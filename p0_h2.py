"""
H2 主实验：动作空间压缩（element_id 分类 vs 坐标回归）

同一观测、同一 backbone、**完全相同的参数量**，唯一变量是动作空间参数化：
  reg : heatmap → soft-argmax → (x, y)    连续
  cls : heatmap → g×g pool   → slot      离散

判据统一：执行点是否落在目标元素 bbox 内。

两个扫描：
  A. 难度扫描（固定 width）：grid ∈ {3,4,5,6}，看 gap 是否随精度要求上升
  B. 参数扫描（最难的 grid）：width ∈ {8,16,32,64}，看 cls 是否更省参数

诊断指标 cell_acc：预测点落在正确格子的比例。
区分「找不到」（cell 也错）与「找到了但不够准」（cell 对、hit 错）——
后者是 H2 的直接证据。

用法：
    python p0_h2.py --quick
    python p0_h2.py --epochs 25
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from p0_env import (
    EnvCfg, make_sample, hit, cell_of, instruction_onehot, LABELS,
)
from p0_models import P0Net

CACHE = Path("p0_cache")
CACHE.mkdir(exist_ok=True)


# ---------- 数据集 ----------

def cfg_for(grid: int, fill: float = 0.70, img_size: int = 160) -> EnvCfg:
    return EnvCfg(grid=grid, img_size=img_size, label_vocab=grid * grid,
                  fill=fill, jitter=4, color_by_label=True)


def precision_demand(cfg: EnvCfg) -> float:
    """精度要求 ≈ 元素线性尺寸 / 图像尺寸。越小越难命中。"""
    cell = cfg.img_size / cfg.grid
    return round((cell * cfg.fill) / cfg.img_size, 4)


def build_dataset(n: int, seed: int, cfg: EnvCfg) -> dict:
    rng = np.random.default_rng(seed)
    S, C = cfg.num_slots, cfg.img_size
    imgs = np.zeros((n, 3, C, C), dtype=np.uint8)
    inst = np.zeros((n, len(LABELS)), dtype=np.float32)
    slots = np.zeros(n, dtype=np.int64)
    xys = np.zeros((n, 2), dtype=np.float32)
    bbox = np.zeros((n, 4), dtype=np.int32)
    centers = np.zeros((n, S, 2), dtype=np.float32)

    for i in range(n):
        s = make_sample(rng, cfg)
        imgs[i] = (s.image * 255.0).astype(np.uint8)
        inst[i] = instruction_onehot(s.instruction, len(LABELS))
        slots[i] = s.target_slot
        xys[i] = s.target_xy
        bbox[i] = s.target_bbox
        for e in s.elements:
            centers[i, e.slot] = (e.center[0] / C, e.center[1] / C)
    return {"images": imgs, "inst": inst, "slots": slots, "xys": xys,
            "bbox": bbox, "centers": centers}


def env_key(cfg) -> str:
    """缓存 key 必须覆盖**所有**影响数据的 env 参数，否则会读到陈旧数据。"""
    return (f"g{cfg.grid}_i{cfg.img_size}_v{cfg.label_vocab}"
            f"_f{cfg.fill}_j{cfg.jitter}_c{int(cfg.color_by_label)}")


def load_or_build(n: int, seed: int, cfg: EnvCfg, tag: str) -> dict:
    path = CACHE / f"{tag}_{env_key(cfg)}_n{n}_{seed}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    d = build_dataset(n, seed, cfg)
    np.savez_compressed(path, **d)
    return d


# ---------- 训练 / 评估 ----------

def to_t(x, device):
    if x.dtype == np.uint8:
        return torch.from_numpy(x).float().div_(255.0).to(device)
    return torch.from_numpy(x).to(device)


def evaluate(model, data, head, cfg, device, batch=512):
    model.eval()
    images = to_t(data["images"], device)
    inst = to_t(data["inst"], device)
    outs = []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch):
            outs.append(model(images[i:i + batch], inst[i:i + batch]).cpu())
    out = torch.cat(outs, 0).numpy()

    n = out.shape[0]
    bbox, centers, tgt = data["bbox"], data["centers"], data["slots"]

    if head == "reg":
        xy = out
        pred_cell = np.array([cell_of(xy[i], cfg) for i in range(n)])
    else:
        pred_cell = out.argmax(-1)
        xy = np.array([centers[i, pred_cell[i]] for i in range(n)])

    hits = np.array([hit(xy[i], bbox[i], cfg) for i in range(n)])
    return {"hit_rate": float(hits.mean()),
            "cell_acc": float((pred_cell == tgt).mean())}


def train_one(width, head, tr, te, cfg, epochs, device,
              lr=3e-3, batch=64, seed=0, stages=3, reg_loss="gauss"):
    """
    reg_loss:
      "coord" — 坐标 MSE（经典但易退化：热力图模糊化 → 塌向图像中心）
      "gauss" — 高斯热力图空间 CE（对 reg 最强的监督，steelman）
    """
    torch.manual_seed(seed)
    model = P0Net(width=width, head=head, num_slots=cfg.num_slots,
                  grid=cfg.grid, num_labels=len(LABELS),
                  in_res=cfg.img_size, stages=stages).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    images = to_t(tr["images"], device)
    inst = to_t(tr["inst"], device)
    if head == "reg" and reg_loss == "coord":
        target = to_t(tr["xys"], device)
    else:
        target = to_t(tr["slots"] if head == "cls" else tr["xys"], device)

    n = images.shape[0]
    spe = max(1, n // batch)
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(spe):
            idx = perm[s * batch:(s + 1) * batch]
            if head == "reg" and reg_loss == "gauss":
                heat = model.heat(images[idx], inst[idx])
                B = heat.shape[0]
                tgt = model.gauss_target(target[idx]).reshape(B, -1)
                loss = -(tgt * F.log_softmax(heat.reshape(B, -1), -1)).sum(-1).mean()
            else:
                o = model(images[idx], inst[idx])
                loss = F.mse_loss(o, target[idx]) if head == "reg" \
                    else F.cross_entropy(o, target[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    m = evaluate(model, te, head, cfg, device)
    m.update({"head": head, "width": width, "grid": cfg.grid,
              "params": model.n_params(), "train_s": round(time.time() - t0, 1),
              "reg_loss": reg_loss if head == "reg" else "-", "seed": seed})
    return m


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    if args.quick:
        n_tr, n_te, epochs, seeds = 1200, 400, 6, 1
        grids, widths, fills = [3, 6], [16, 32], [0.70, 0.35]
    else:
        n_tr, n_te, epochs, seeds = 4000, 1000, args.epochs, args.seeds
        grids, widths, fills = [3, 4, 5, 6, 8], [8, 16, 32, 64], [0.75, 0.55, 0.40, 0.28]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} train={n_tr} test={n_te} epochs={epochs} seeds={seeds}")

    tag = "q" if args.quick else "m"
    ref_w, ref_g = 32, 6
    results = []

    def run(cfg, width, label, modes):
        tr = load_or_build(n_tr, 0, cfg, f"tr{tag}")
        te = load_or_build(n_te, 999, cfg, f"te{tag}")
        for head, rl in modes:
            ms = [train_one(width, head, tr, te, cfg, epochs, device,
                            seed=s, reg_loss=rl) for s in range(seeds)]
            hit_m = float(np.mean([m["hit_rate"] for m in ms]))
            cell_m = float(np.mean([m["cell_acc"] for m in ms]))
            hit_sd = float(np.std([m["hit_rate"] for m in ms]))
            r = dict(ms[0])
            r.update({"hit_rate": hit_m, "cell_acc": cell_m, "hit_sd": hit_sd,
                      "fill": cfg.fill, "precision": precision_demand(cfg),
                      "tag": label, "head": head, "reg_loss": rl,
                      "width": width, "grid": cfg.grid})
            print(f"  {label} {head}/{rl}: hit={hit_m:.4f}±{hit_sd:.4f} "
                  f"cell={cell_m:.4f} params={r['params']:,}")
            results.append(r)
        del tr, te

    # --- 扫描 A：精度要求 ---
    print(f"\n########## 扫描 A：精度要求 (grid={ref_g}, width={ref_w}) ##########")
    for f in fills:
        c = cfg_for(ref_g, fill=f)
        print(f"[fill={f} precision={precision_demand(c):.4f}]")
        run(c, ref_w, f"fill={f}", [("cls", "-"), ("reg", "coord")])

    # --- 扫描 B：格子数 ---
    print(f"\n########## 扫描 B：格子数 (width={ref_w}, fill=0.40) ##########")
    for g in grids:
        c = cfg_for(g, fill=0.40)
        print(f"[grid={g} slots={c.num_slots}]")
        run(c, ref_w, f"grid={g}", [("cls", "-"), ("reg", "coord")])

    # --- 扫描 C：参数量（最难配置）---
    hard = cfg_for(grids[-1], fill=fills[-1])
    print(f"\n########## 扫描 C：参数量 (grid={grids[-1]}, fill={fills[-1]}, "
          f"precision={precision_demand(hard):.4f}) ##########")
    for w in widths:
        run(hard, w, f"w={w}", [("cls", "-"), ("reg", "gauss"), ("reg", "coord")])

    # --- 汇总 ---
    def show(rs, ref, title):
        print(f"\n{'-'*84}\n{title}\n{'-'*84}")
        print(f"{'cfg':>10}{'params':>10}{'cls hit':>10}{'reg hit':>10}"
              f"{'Δhit':>9}{'cls cell':>10}{'reg cell':>10}")
        keys = list(dict.fromkeys(r[ref] for r in rs))
        for k in keys:
            c = next((r for r in rs if r["head"] == "cls" and r[ref] == k
                      and not str(r["reg_loss"]).startswith("coord")), None)
            r_ = next((r for r in rs if r["head"] == "reg" and r["reg_loss"] == "coord"
                       and r[ref] == k), None)
            if not c or not r_:
                continue
            print(f"{str(k):>10}{c['params']:>10,}{c['hit_rate']:>10.4f}"
                  f"{r_['hit_rate']:>10.4f}{c['hit_rate']-r_['hit_rate']:>+9.4f}"
                  f"{c['cell_acc']:>10.4f}{r_['cell_acc']:>10.4f}")

    show([r for r in results if r["tag"].startswith("fill")], "tag", "扫描 A：精度要求")
    show([r for r in results if r["tag"].startswith("grid")], "tag", "扫描 B：格子数")
    show([r for r in results if r["tag"].startswith("w=")], "tag", "扫描 C：参数量")

    # reg 两种损失的对比（scan C）
    print(f"\n{'-'*84}\nreg 损失函数对比（扫描 C，检验 H2 是否只是损失函数问题）\n{'-'*84}")
    print(f"{'cfg':>10}{'params':>10}{'reg gauss':>12}{'reg coord':>12}{'cls':>10}")
    for w in widths:
        g_ = next((r for r in results if r["tag"] == f"w={w}"
                   and r["head"] == "reg" and r["reg_loss"] == "gauss"), None)
        c_ = next((r for r in results if r["tag"] == f"w={w}"
                   and r["head"] == "reg" and r["reg_loss"] == "coord"), None)
        k_ = next((r for r in results if r["tag"] == f"w={w}" and r["head"] == "cls"), None)
        if g_ and c_ and k_:
            print(f"{str(w):>10}{k_['params']:>10,}{g_['hit_rate']:>12.4f}"
                  f"{c_['hit_rate']:>12.4f}{k_['hit_rate']:>10.4f}")

    out = {"device": str(device), "n_train": n_tr, "n_test": n_te,
           "epochs": epochs, "seeds": seeds, "results": results}
    Path("p0_h2_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 p0_h2_results.json")


if __name__ == "__main__":
    main()
