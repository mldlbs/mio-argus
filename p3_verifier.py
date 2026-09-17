"""
P3：结果验证（H3 唯一存活的形式）。

问题
----
P2 证明：重新接地的 verifier 与策略同构，无法带来增益。
那么**结果验证**呢？
    V(旧屏幕, 新屏幕) → 进度是否推进？

三组对照（隔离"验证"与"多试几次"两种效应）：

  A. π 单次          每步只出 top-1，无验证、无重试
  B. π + 盲目重试   每步最多 R 次，但重试不依赖任何判断（按 π 分布采样）
  C. π + V 门控重试 每步最多 R 次，V 判断是否推进，未推进才重试

预测：C > B ≥ A；且 C 相对 A 的增益随 H 增长（每步失误被复利放大）。

用法:
    python p3_verifier.py --quick
    python p3_verifier.py --epochs 25 --v-epochs 15
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from p0_env import EnvCfg, LABELS, instruction_onehot
from p0_models import P0Net
from p0_h2 import to_t
from p3_env import render_ui, make_episode, fresh_labels

IMG_SIZE = 160
GRID = 6
CACHE = Path("p0_cache")


def cfg_p3() -> EnvCfg:
    return EnvCfg(grid=GRID, img_size=IMG_SIZE, label_vocab=GRID * GRID,
                  fill=0.40, jitter=4, color_by_label=True)


# ---------- π 的数据（含进度条，进度随机以使 π 学会忽略它）----------

def build_pi_data(n, seed, cfg):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    inst = np.zeros((n, len(LABELS)), dtype=np.float32)
    slots = np.zeros(n, dtype=np.int64)
    for i in range(n):
        labels = fresh_labels(cfg, rng)
        slot = int(rng.integers(0, cfg.num_slots))
        H = int(rng.choice([1, 2, 4, 8]))
        prog = int(rng.integers(0, H + 1))
        wrong = [False] * cfg.num_slots
        done = [False] * cfg.num_slots
        img, _ = render_ui(labels, prog, H, wrong, done, cfg, geom_seed=i)
        imgs[i] = (img * 255).astype(np.uint8)
        inst[i] = instruction_onehot(f"click {labels[slot]}", len(LABELS))
        slots[i] = slot
    return {"images": imgs, "inst": inst, "slots": slots,
            "xys": np.zeros((n, 2), dtype=np.float32)}


def train_pi(width, tr, epochs, device, seed=0, lr=3e-3, batch=64):
    torch.manual_seed(seed)
    m = P0Net(width=width, head="cls", num_slots=GRID * GRID, grid=GRID,
              num_labels=len(LABELS), in_res=IMG_SIZE, stages=3).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    images = to_t(tr["images"], device)
    inst = to_t(tr["inst"], device)
    tgt = torch.from_numpy(tr["slots"]).to(device)
    n = images.shape[0]; spe = max(1, n // batch)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(spe):
            idx = perm[s * batch:(s + 1) * batch]
            loss = F.cross_entropy(m(images[idx], inst[idx]), tgt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def pi_acc(m, data, device, batch=256):
    m.eval()
    images = to_t(data["images"], device); inst = to_t(data["inst"], device)
    ok = 0
    with torch.no_grad():
        for i in range(0, images.shape[0], batch):
            p = m(images[i:i + batch], inst[i:i + batch]).argmax(-1).cpu().numpy()
            ok += int((p == data["slots"][i:i + batch]).sum())
    return ok / data["images"].shape[0]


# ---------- 结果验证器 ----------

class OutcomeVerifier(nn.Module):
    """输入 (旧帧, 新帧) 6 通道 → 进度是否推进。刻意做小。"""

    def __init__(self, width=8, in_res=IMG_SIZE):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(6, w, 3, stride=2, padding=1), nn.BatchNorm2d(w), nn.ReLU(),
            nn.Conv2d(w, w * 2, 3, stride=2, padding=1), nn.BatchNorm2d(w * 2), nn.ReLU(),
            nn.Conv2d(w * 2, w * 4, 3, stride=2, padding=1), nn.BatchNorm2d(w * 4), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(w * 4, 1),
        )

    def forward(self, before, after):
        return self.net(torch.cat([before, after], dim=1)).squeeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_v_data(n, seed, cfg):
    """随机动作 → 记录 (before, after, advanced)。"""
    rng = np.random.default_rng(seed)
    before = np.zeros((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    after = np.zeros((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    label = np.zeros(n, dtype=np.float32)

    for i in range(n):
        labels = fresh_labels(cfg, rng)
        H = int(rng.choice([1, 2, 4, 8]))
        prog = int(rng.integers(0, H))
        wrong = [bool(rng.random() < 0.2) for _ in range(cfg.num_slots)]
        done = [bool(rng.random() < 0.2) for _ in range(cfg.num_slots)]
        target = labels[int(rng.integers(0, cfg.num_slots))]

        b, _ = render_ui(labels, prog, H, wrong, done, cfg, geom_seed=i)
        before[i] = (b * 255).astype(np.uint8)

        # 50% 正确动作，50% 错误动作
        if rng.random() < 0.5:
            slot = labels.index(target)
            adv = 1.0
            new_wrong, new_done, new_prog = list(wrong), list(done), prog + 1
            new_done[slot] = True
        else:
            slot = int(rng.integers(0, cfg.num_slots))
            if slot == labels.index(target):
                slot = (slot + 1) % cfg.num_slots
            adv = 0.0
            new_wrong, new_done, new_prog = list(wrong), list(done), prog
            new_wrong[slot] = True

        a, _ = render_ui(labels, new_prog, H, new_wrong, new_done, cfg, geom_seed=i)
        after[i] = (a * 255).astype(np.uint8)
        label[i] = adv

    return {"before": before, "after": after, "label": label}


def train_v(width, data, epochs, device, seed=0, lr=2e-3, batch=64):
    torch.manual_seed(seed)
    v = OutcomeVerifier(width=width).to(device)
    opt = torch.optim.Adam(v.parameters(), lr=lr)
    B = to_t(data["before"], device); A = to_t(data["after"], device)
    y = torch.from_numpy(data["label"]).to(device)
    n = B.shape[0]; spe = max(1, n // batch)
    v.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(spe):
            idx = perm[s * batch:(s + 1) * batch]
            loss = F.binary_cross_entropy_with_logits(v(B[idx], A[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return v


def v_eval(v, data, device, batch=256):
    v.eval()
    B = to_t(data["before"], device); A = to_t(data["after"], device)
    y = data["label"]
    pred = []
    with torch.no_grad():
        for i in range(0, B.shape[0], batch):
            pred.append((torch.sigmoid(v(B[i:i + batch], A[i:i + batch])) > 0.5)
                        .float().cpu().numpy())
    p = np.concatenate(pred)
    tpr = float(p[y == 1].mean()) if (y == 1).any() else float("nan")
    fpr = float(p[y == 0].mean()) if (y == 0).any() else float("nan")
    return {"acc": float((p == y).mean()), "tpr": tpr, "fpr": fpr,
            "tnr": 1.0 - fpr}


# ---------- rollout ----------

def pi_probs(m, img, label, device):
    m.eval()
    with torch.no_grad():
        it = torch.from_numpy(img).unsqueeze(0).to(device)
        oh = torch.from_numpy(instruction_onehot(f"click {label}", len(LABELS)))
        oh = oh.unsqueeze(0).to(device)
        return torch.softmax(m(it, oh), -1).cpu().numpy()[0]


def v_advanced(v, before, after, device):
    v.eval()
    with torch.no_grad():
        b = torch.from_numpy(before).unsqueeze(0).to(device)
        a = torch.from_numpy(after).unsqueeze(0).to(device)
        return float(torch.sigmoid(v(b, a)).item()) > 0.5


def rollout(pi, v, cfg, device, H, rho, n_tasks, seed,
            max_retry=3, mode="pi"):
    """
    mode:
      "pi"     每步只出 top-1，无验证、无重试（失败即任务失败）
      "v"      V 门控重试：V 判断进度是否推进，未推进则试下一个候选
      "oracle" 完美验证器重试（上界）
    """
    S = cfg.num_slots
    rng = np.random.default_rng(seed)
    task_ok = []

    for ti in range(n_tasks):
        steps = make_episode(cfg, rng, H, rho)
        labels = list(steps[0].labels)
        wrong = [False] * S
        done = [False] * S
        progress = 0
        geom = ti * 100 + 0          # 布局版本 → 决定元素几何
        img, _ = render_ui(labels, progress, H, wrong, done, cfg, geom_seed=geom)

        success = True
        for t in range(H):
            if rho > 0 and t > 0 and rng.random() < rho:
                labels = fresh_labels(cfg, rng)
                wrong = [False] * S
                done = [False] * S
                geom = ti * 100 + t
                img, _ = render_ui(labels, progress, H, wrong, done, cfg,
                                   geom_seed=geom)

            target = steps[t].target
            tgt_slot = labels.index(target)
            p = pi_probs(pi, img, target, device)
            order = list(np.argsort(-p))

            R = 1 if mode == "pi" else max_retry
            tried = set()

            for _ in range(R):
                cand = [c for c in order if c not in tried]
                slot = int((cand or order)[0])
                tried.add(slot)

                correct = (slot == tgt_slot)
                nw, nd = list(wrong), list(done)
                if correct:
                    nd[slot] = True
                else:
                    nw[slot] = True
                nprog = progress + (1 if correct else 0)
                nimg, _ = render_ui(labels, nprog, H, nw, nd, cfg, geom_seed=geom)

                if mode == "pi":
                    wrong, done, progress, img = nw, nd, nprog, nimg
                    break

                accept = correct if mode == "oracle" \
                    else v_advanced(v, img, nimg, device)

                if accept:
                    wrong, done, progress, img = nw, nd, nprog, nimg
                    break
                else:
                    wrong, done, img = nw, nd, nimg   # 进度不变

            # 成功判据必须是**进度真的推进到 t+1**，
            # 而不是"verifier 接受了"——否则一个恒说接受的 V 能刷满任务。
            if progress != t + 1:
                success = False
                break

        task_ok.append(success)

    return {"task_success": float(np.mean(task_ok))}


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--v-epochs", type=int, default=15)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--v-width", type=int, default=8)
    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--tasks", type=int, default=300)
    args = ap.parse_args()

    epochs = 8 if args.quick else args.epochs
    v_ep = 8 if args.quick else args.v_epochs
    n_tr, n_te = (1500, 500) if args.quick else (4000, 1000)
    Hs = [1, 2, 4] if args.quick else [1, 2, 4, 8]
    rhos = [1.0] if args.quick else [0.3, 1.0]
    n_tasks = 80 if args.quick else args.tasks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = cfg_p3()
    print(f"device={device} H={Hs} rho={rhos} retry={args.retry} tasks={n_tasks}")

    # π
    ptr = build_pi_data(n_tr, 0, cfg)
    pte = build_pi_data(n_te, 999, cfg)
    pi = train_pi(args.width, ptr, epochs, device)
    acc = pi_acc(pi, pte, device)
    print(f"[π] params={pi.n_params() if hasattr(pi,'n_params') else sum(p.numel() for p in pi.parameters()):,}"
          f"  single-step acc={acc:.4f}")

    # V
    t0 = time.time()
    vtr = build_v_data(n_tr, 1, cfg)
    vte = build_v_data(n_te, 998, cfg)
    V = train_v(args.v_width, vtr, v_ep, device)
    vp = V.n_params()
    vm = v_eval(V, vte, device)
    print(f"[V] params={vp:,}  acc={vm['acc']:.4f} "
          f"(tpr={vm['tpr']:.4f} tnr={vm['tnr']:.4f})  训练 {time.time()-t0:.0f}s")

    rows = []
    print(f"\n{'rho':>5}{'H':>4}{'pi':>9}{'pi+V':>9}{'pi+oracle':>11}"
          f"{'V gain':>9}{'oracle gain':>13}")
    print("-" * 58)
    for rho in rhos:
        for H in Hs:
            sd = 2000 + H * 7 + int(rho * 100)
            a = rollout(pi, None, cfg, device, H, rho, n_tasks, sd,
                        args.retry, "pi")["task_success"]
            c = rollout(pi, V, cfg, device, H, rho, n_tasks, sd,
                        args.retry, "v")["task_success"]
            o = rollout(pi, None, cfg, device, H, rho, n_tasks, sd,
                        args.retry, "oracle")["task_success"]
            rows.append({"rho": rho, "H": H, "pi": a, "pi_V": c,
                         "pi_oracle": o})
            print(f"{rho:>5.1f}{H:>4}{a:>9.4f}{c:>9.4f}{o:>11.4f}"
                  f"{c-a:>+9.4f}{o-a:>+13.4f}")
        print()

    out = {"device": str(device), "grid": GRID, "pi_width": args.width,
           "v_width": args.v_width, "retry": args.retry,
           "pi_params": sum(p.numel() for p in pi.parameters()),
           "v_params": vp, "pi_single_step_acc": acc, "v_metrics": vm,
           "n_train": n_tr, "n_test": n_te, "n_tasks": n_tasks,
           "Hs": Hs, "rhos": rhos, "rows": rows}
    Path("p3_verifier_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 p3_verifier_results.json")


if __name__ == "__main__":
    main()
