"""
QGIS W11：验证的真实成本 —— 从"成功率"到"正确产物/分钟"与盈亏平衡点

W10 证明了：依赖链 + 可观测结果下，full-V 把成功率从 0.083 拉到 0.917。
但 W10 里**验证是免费的**（重掷不花任何东西），所以那个增益无法回答
"V 值不值" —— 这正是 QGIS_FINDINGS §6 标为最高优先的遗留项。

本轮给每一步动作与每一次验证都记真实耗时，然后固定时间预算跑任务流：
    指标 = 正确产物 / 分钟（success per minute），而不是成功率。
对比 4 种模式：
    nov     不验证
    full    三步都验证（S1/S2 截图 V + S3 读文件）
    cheap   只验证 S1（截图）与 S3（读文件）——省掉最弱的 V(S2)
    full4x  同 full，但每次截图验证额外睡 3 倍时间（模拟更慢的 verifier）
再解析求**盈亏平衡**：verifier 单次成本增大到多少，V 就不划算了。

用法: python experiments/qgis/qgis_bridge/run_w11.py
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402

from PIL import Image  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CW, CH = 256, 128
CROP = (340, 120, 1940, 1220)
GEOJSON = HERE / "data" / "sel.geojson"
SCENE = "regions"
RIGHT = {"North", "South"}
Q = 0.5
MAX_RETRY = 3
BUDGET = 150.0          # 每个模式的任务流时间预算（秒）


def truth():
    return call({"op": "truth"})["truth"]


def selected():
    t = truth()
    return t["layers"][0]["selected"] if t.get("layers") else None


def extent_x0():
    return truth()["extent"][0]


def grab(path):
    call({"op": "shot", "path": str(path)})
    img = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = CROP
    img = img.crop((x0, y0, min(img.width, x1), min(img.height, y1)))
    return np.asarray(img.resize((CW, CH), Image.BILINEAR))


def do_s1(correct):
    expr = '"area" > 300' if correct else '"area" <= 300'
    return call({"op": "select_expr", "expr": expr, "layer": SCENE})["ok"]


def do_s2(correct):
    if correct:
        return call({"op": "zoom_to_selection"})["ok"]
    return call({"op": "act", "kind": "action", "text": "Zoom In"})["ok"]


def do_s3(correct):
    return call({"op": "export", "path": str(GEOJSON),
                 "only_selected": bool(correct), "layer": SCENE})["ok"]


def artifact_names():
    try:
        d = json.load(open(GEOJSON, encoding="utf-8"))
        return {f["properties"]["name"] for f in d["features"]}
    except Exception:
        return set()


class V(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(nn.Conv2d(6, 16, 3, 2, 1), nn.ReLU(),
                            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2))
    def forward(s, x):
        return s.f(x)


def train_v(step, rng, ref_x0, n=100, epochs=40):
    tmp = HERE / f"_w11_{step}.png"
    data, labels = [], []
    for i in range(n):
        call({"op": "reset", "scene": SCENE})
        time.sleep(0.15)
        if step == "s2":
            do_s1(True)
            time.sleep(0.2)
        before = grab(tmp)
        (do_s1 if step == "s1" else do_s2)(i % 2 == 0)
        time.sleep(0.25)
        after = grab(tmp)
        data.append(np.concatenate([before, after], -1))
        ok = (selected() == 2) if step == "s1" else (abs(extent_x0()-ref_x0) < 1.0)
        labels.append(int(ok))
    X = torch.tensor(np.stack(data)).permute(0, 3, 1, 2).float()/255.0
    y = torch.tensor(labels)
    perm = rng.permutation(len(labels)); cut = int(len(labels)*0.75)
    tr, te = perm[:cut], perm[cut:]
    torch.manual_seed(0)
    m = V(); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for ep in range(epochs):
        p = torch.randperm(len(tr))
        for i in range(0, len(tr), 32):
            idx = p[i:i+32]
            loss = F.cross_entropy(m(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        prob = F.softmax(m(X), -1)[:, 1].numpy()
    order = np.argsort(prob); ranks = np.empty(len(prob)); ranks[order] = np.arange(1, len(prob)+1)
    pos, neg = y.numpy() == 1, y.numpy() == 0
    auc = float((ranks[pos].sum() - pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum()))
    pv = prob[te]; yv = y[te].numpy()
    cand = np.unique(np.concatenate([[0.0, 1.0], (np.sort(pv)[:-1]+np.sort(pv)[1:])/2]))
    bt, bb = 0.5, -1
    for t in cand:
        pr = (pv > t).astype(int)
        tp = ((pr == 1) & (yv == 1)).sum(); fn = ((pr == 0) & (yv == 1)).sum()
        tn = ((pr == 0) & (yv == 0)).sum(); fp = ((pr == 1) & (yv == 0)).sum()
        bal = 0.5*(tp/max(1, tp+fn) + tn/max(1, tn+fp))
        if bal > bb:
            bb, bt = bal, float(t)
    print(f"V({step})：AUC={auc:.3f} 校准 acc={float(((pv > bt).astype(int) == yv).mean()):.3f}")
    return m, bt


def eval_v(m, thr, before, after, extra_sleep=0.0):
    if extra_sleep > 0:
        time.sleep(extra_sleep)
    x = torch.tensor(np.concatenate([before, after], -1))[None]
    x = x.permute(0, 3, 1, 2).float()/255.0
    with torch.no_grad():
        return float(F.softmax(m(x), -1)[0, 1]) > thr


def do_step(name, correct, fn):
    """执行一步（不打勾），返回 (ok, 耗时)。"""
    t0 = time.time()
    fn(correct)
    return True, time.time() - t0


def measure_costs(vs, tmp, rng):
    """逐一实测各组件耗时（中位数，n=7）。"""
    c = {}

    def med(fn, n=7):
        xs = []
        for _ in range(n):
            xs.append(fn())
        return statistics.median(xs)

    c["reset"] = med(lambda: (lambda t0: (call({"op": "reset", "scene": SCENE}),
                                          time.time()-t0)[1])(time.time()))
    c["s1"] = med(lambda: do_step("s1", True, do_s1)[1])
    c["s2"] = med(lambda: do_step("s2", True, do_s2)[1])
    c["s3"] = med(lambda: do_step("s3", True, do_s3)[1])
    c["v_shot"] = med(lambda: (lambda t0: (grab(tmp), time.time()-t0)[1])(time.time()))
    m, thr = vs["s1"]
    b = grab(tmp); a = grab(tmp)
    c["v_forward"] = med(lambda: (lambda t0: (eval_v(m, thr, b, a), time.time()-t0)[1])(time.time()))
    c["v_file"] = med(lambda: (lambda t0: (artifact_names(), time.time()-t0)[1])(time.time()))
    return c


MODES = ("nov", "full", "cheap", "full2x", "full3x")
VMULT = {"full2x": 2.0, "full3x": 3.0}      # 验证直接成本（截图×2+前向）的放大倍数


def _v_extra(mode, c):
    """模拟更贵的 verifier：直接成本（2 次截图 + 前向）乘 VMULT[mode]。"""
    mult = VMULT.get(mode, 1.0)
    return (mult - 1.0) * (2.0*c["v_shot"] + c["v_forward"])


def stream(mode, vs, rng, tmp, c, budget=BUDGET):
    """固定时间预算内连续跑任务，返回 (正确数, 总数, 用时)。"""
    t0 = time.time(); n = 0; ok_n = 0
    extra = _v_extra(mode, c)
    while time.time() - t0 < budget:
        n += 1
        call({"op": "reset", "scene": SCENE})
        # S1
        done = False
        for _ in range(1 + MAX_RETRY):
            correct = rng.random() < Q
            verify = mode in ("full", "cheap", "full2x", "full3x")
            if not verify:
                do_s1(correct)
                done = True
                break
            before = grab(tmp)          # 必须在动作之前
            do_s1(correct)
            time.sleep(0.2)
            after = grab(tmp)
            if eval_v(vs["s1"][0], vs["s1"][1], before, after, extra):
                done = True
                break
        if not done:
            continue
        # S2
        if mode in ("full", "full2x", "full3x"):
            time.sleep(0.05)
            for _ in range(1 + MAX_RETRY):
                correct = rng.random() < Q
                before = grab(tmp)
                do_s2(correct)
                time.sleep(0.2)
                after = grab(tmp)
                if eval_v(vs["s2"][0], vs["s2"][1], before, after, extra):
                    break
        else:
            do_s2(rng.random() < Q)
            time.sleep(0.05)
        # S3
        for _ in range(1 + MAX_RETRY):
            correct = rng.random() < Q
            do_s3(correct)
            if mode == "nov":
                break
            if artifact_names() == RIGHT:
                break
        ok = artifact_names() == RIGHT and abs(extent_x0() - REF_X0) < 1.0
        ok_n += int(ok)
    return ok_n, n, time.time() - t0


REF_X0 = None


def main():
    global REF_X0
    rng = np.random.default_rng(0)
    call({"op": "win", "maximize": True})
    tmp = HERE / "_w11_tmp.png"

    call({"op": "reset", "scene": SCENE})
    time.sleep(0.2)
    do_s1(True); time.sleep(0.2); call({"op": "zoom_to_selection"}); time.sleep(0.4)
    REF_X0 = extent_x0()
    print(f"参考 extent_x0 = {REF_X0:.2f}")

    vs = {"s1": train_v("s1", rng, REF_X0), "s2": train_v("s2", rng, REF_X0)}
    c = measure_costs(vs, tmp, rng)
    print("\n组件耗时（中位数，ms）:")
    for k, v in c.items():
        print(f"  {k:>10}: {v*1000:7.1f}")
    c_v = c["v_shot"] + c["v_forward"]

    print(f"\n时间预算 {BUDGET:.0f}s/模式  q={Q}  任务 = S1→S2→S3 依赖链")
    print(f"{'mode':>7}{'正确/总数':>12}{'成功率':>9}{'产物/分钟':>10}")
    print("-" * 38)
    out = {}
    for mode in MODES:
        ok_n, n, el = stream(mode, vs, rng, tmp, c)
        row = {"ok": ok_n, "n": n, "rate": ok_n/n if n else 0.0,
               "per_min": ok_n/(el/60), "elapsed": el}
        out[mode] = row
        print(f"{mode:>7}{f'{ok_n}/{n}':>12}{row['rate']:>9.3f}{row['per_min']:>10.2f}")
    print("-" * 38)

    # 盈亏平衡：verifier 每次成本放大 m 倍，full 的产物/分钟何时降到 nov 之下
    r_nov, r_fl = out["nov"]["per_min"], out["full"]["per_min"]
    print(f"\n盈亏平衡（解析）：nov = {r_nov:.2f}/min，full = {r_fl:.2f}/min")
    if r_fl > 0:
        # full 每集验证次数 ≈ 每步 (1 + 重掷) 次；用实测估计：总时间中验证占比
        nv = out["full"]["n"]
        # 反推每集验证次数：由 stream 的额外开销近似（v_shot+v_forward+0.2s sleep）恒定
        per_ep_check = 3 * (1/ (1 - (1-Q)))   # 上界估计：3 步 × 平均 2 次尝试
        adv = (r_fl - r_nov) / max(r_fl, 1e-9)
        print(f"  若 verifier 单次成本从 {c_v*1000:.0f} ms 放大到 X 倍，full 的优势线性缩水；"
              f"按现有数据，放大 ~{max(1.0, 1/(1-adv)):.1f}× 时二者持平"
              if adv < 1 else "  full 全面占优（放大后仍不掉到 nov 之下）")

    (HERE / "w11_results.json").write_text(json.dumps(
        {"scene": SCENE, "q": Q, "budget": BUDGET, "costs_ms": {k: v*1000 for k, v in c.items()},
         "modes": out, "ref_x0": REF_X0}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w11_results.json")


if __name__ == "__main__":
    main()
