"""
P1：检验 H1 —— 重观察能否替代参数，且这个替代价值是否随 horizon 增长。

设计
----
多步任务：任务有 H 步。每一步 t 有一个布局 l_t 和一个目标标签 L_t，
agent 必须点击"实际布局 l_t 中标签为 L_t 的格子"。

布局演化：l_t = l_{t-1}，概率 ρ；否则重新随机（ρ=0 静态，ρ=1 全随机）。

Agent 的观察频率由间隔 k 决定：
    t % k == 0  → 看到当前真实屏幕
    否则        → 沿用上次观察到的（可能已过期的）屏幕

因此 H1 的可测形式是：

    成功率 = f(ρ, k, H)

预测：
  - k=1（每步重观察）→ 成功率与 ρ、H 基本无关
  - k=H（只观察一次）→ 成功率随 ρ 和 H 快速崩塌
  - 参数增加**不能**弥补观察缺失（信息不在参数里）

用法:
    python p1_horizon.py --quick
    python p1_horizon.py --epochs 25
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from p0_env import (EnvCfg, Element, render, hit, instruction_onehot,
                    PALETTE, LABEL_TO_ID, LABELS)
from p0_models import P0Net
from p0_h2 import load_or_build, train_one, cfg_for

CACHE = Path("p0_cache")
IMG_SIZE = 160
GRID = 6


# ---------- 布局渲染（与 p0_env.make_sample 相同的尺寸逻辑，但标签由外部给定）----------

def build_elements(labels, cfg: EnvCfg, rng) -> list:
    els = []
    for slot in range(cfg.num_slots):
        bx1, by1, bx2, by2 = cfg.slot_bbox(slot)
        cw, ch = bx2 - bx1, by2 - by1
        ew = max(6, int(cw * cfg.fill) - int(rng.integers(0, 5)))
        eh = max(6, int(ch * cfg.fill) - int(rng.integers(0, 5)))
        ex1 = int(np.clip(bx1 + rng.integers(0, cfg.jitter + 1), 0, cfg.img_size - 2))
        ey1 = int(np.clip(by1 + rng.integers(0, cfg.jitter + 1), 0, cfg.img_size - 2))
        ex2 = int(np.clip(ex1 + ew, ex1 + 1, cfg.img_size - 1))
        ey2 = int(np.clip(ey1 + eh, ey1 + 1, cfg.img_size - 1))
        label = labels[slot]
        color = PALETTE[LABEL_TO_ID[label] % len(PALETTE)]
        els.append(Element(slot, (ex1, ey1, ex2, ey2), label, "button", color))
    return els


def render_layout(labels, cfg, rng):
    els = build_elements(labels, cfg, rng)
    return render(els, cfg), els


def fresh_layout(cfg, rng):
    vocab = LABELS[:cfg.label_vocab]
    idx = rng.choice(len(vocab), size=cfg.num_slots, replace=False)
    return [vocab[int(i)] for i in idx]


def make_task(cfg, rng, H, rho):
    """返回 [(labels_t, L_t), ...]，含布局演化。"""
    layout = fresh_layout(cfg, rng)
    steps = []
    for t in range(H):
        if t > 0 and rng.random() < rho:
            layout = fresh_layout(cfg, rng)
        L_t = layout[int(rng.integers(0, cfg.num_slots))]
        steps.append((list(layout), L_t))
    return steps


# ---------- 单步预测 ----------

def predict(model, head, img_chw, label, cfg, device):
    """img_chw: (3,H,W) float32 [0,1] → 返回 ('slot', s) 或 ('xy', (x,y))。"""
    with torch.no_grad():
        it = torch.from_numpy(img_chw).unsqueeze(0).to(device)
        oh = torch.from_numpy(instruction_onehot(f"click {label}", len(LABELS)))
        oh = oh.unsqueeze(0).to(device)
        o = model(it, oh).cpu().numpy()[0]
    if head == "cls":
        return "slot", int(np.argmax(o))
    return "xy", (float(o[0]), float(o[1]))


# ---------- rollout ----------

def rollout(model, head, cfg, device, H, rho, k, n_tasks, seed=0):
    rng = np.random.default_rng(seed)
    per_step, task_ok = [], []

    for _ in range(n_tasks):
        task = make_task(cfg, rng, H, rho)

        # 布局未变 → 复用同一张屏幕（真实 UI 静态时像素完全一致）。
        # 若每步重渲染，jitter 会变化，造成"布局没变但观察失效"的伪影。
        screens, prev = [], None
        for labels, _ in task:
            if prev is not None and labels == prev:
                screens.append(screens[-1])
            else:
                screens.append(render_layout(labels, cfg, rng))
            prev = labels

        obs_idx = None
        step_ok = []
        for t, (labels_t, L_t) in enumerate(task):
            actual_els = screens[t][1]

            if (t % k == 0) or (obs_idx is None):
                obs_idx = t
            obs_img = screens[obs_idx][0]

            kind, pred = predict(model, head, obs_img, L_t, cfg, device)

            if kind == "slot":
                ok = (labels_t[pred] == L_t)          # 点击落在**实际**布局上
            else:
                tgt = next(e for e in actual_els if e.label == L_t)
                ok = bool(hit(pred, tgt.bbox, cfg))

            step_ok.append(bool(ok))
            per_step.append(bool(ok))

        task_ok.append(all(step_ok))

    return {"task_success": float(np.mean(task_ok)),
            "step_acc": float(np.mean(per_step))}


def train_single_step(width, head, reg_loss, tr, te, cfg, epochs, device,
                      seed=0, lr=3e-3, batch=64):
    """训练单步策略，返回 (model, metrics)。"""
    from p0_h2 import to_t, evaluate
    import torch.nn.functional as F

    torch.manual_seed(seed)
    model = P0Net(width=width, head=head, num_slots=cfg.num_slots,
                  grid=cfg.grid, num_labels=len(LABELS),
                  in_res=cfg.img_size, stages=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    images = to_t(tr["images"], device)
    inst = to_t(tr["inst"], device)
    target = to_t(tr["xys"] if head == "reg" else tr["slots"], device)

    n = images.shape[0]
    spe = max(1, n // batch)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(spe):
            idx = perm[s * batch:(s + 1) * batch]
            o = model(images[idx], inst[idx])
            loss = F.mse_loss(o, target[idx]) if head == "reg" \
                else F.cross_entropy(o, target[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    metrics = evaluate(model, te, head, cfg, device)
    metrics.update({"head": head, "params": model.n_params()})
    return model, metrics


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--tasks", type=int, default=300)
    args = ap.parse_args()

    epochs = 8 if args.quick else args.epochs
    n_tr, n_te = (1500, 500) if args.quick else (4000, 1000)
    Hs = [1, 2, 4] if args.quick else [1, 2, 4, 8]
    rhos = [0.0, 1.0] if args.quick else [0.0, 0.3, 1.0]
    n_tasks = 80 if args.quick else args.tasks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} grid={GRID} width={args.width} epochs={epochs} "
          f"Hs={Hs} rhos={rhos} tasks/config={n_tasks}")
    tag = "q" if args.quick else "m"

    cfg = cfg_for(GRID, fill=0.40)
    tr = load_or_build(n_tr, 0, cfg, f"tr{tag}")
    te = load_or_build(n_te, 999, cfg, f"te{tag}")

    # 训练单步策略（训练时每步都能看到新鲜屏幕）
    models, single = {}, {}
    for head, rl in [("cls", "-"), ("reg", "coord")]:
        m, mt = train_single_step(args.width, head, rl, tr, te, cfg,
                                  epochs, device, seed=0)
        print(f"[train] {head}: single-step hit={mt['hit_rate']:.4f} "
              f"params={mt['params']:,}")
        models[head], single[head] = m, mt

    rows = []
    print(f"\n{'rho':>5}{'k':>4}{'H':>4}{'cls task':>10}{'reg task':>10}"
          f"{'cls step':>10}{'reg step':>10}")
    print("-" * 60)
    for rho in rhos:
        for H in Hs:
            for k in sorted(set([1, 2, 4, H])):
                if k > H:
                    continue
                r = {}
                for head in ("cls", "reg"):
                    # seed 不依赖 k：保证不同观察间隔跑的是**同一批任务**，可严格比较
                    r[head] = rollout(models[head], head, cfg, device,
                                      H, rho, k, n_tasks,
                                      seed=1000 + H * 7 + int(rho * 100))
                rows.append({"rho": rho, "H": H, "k": k,
                             "cls_task": r["cls"]["task_success"],
                             "reg_task": r["reg"]["task_success"],
                             "cls_step": r["cls"]["step_acc"],
                             "reg_step": r["reg"]["step_acc"]})
                print(f"{rho:>5.1f}{k:>4}{H:>4}{r['cls']['task_success']:>10.4f}"
                      f"{r['reg']['task_success']:>10.4f}"
                      f"{r['cls']['step_acc']:>10.4f}{r['reg']['step_acc']:>10.4f}")
            print()

    out = {"device": str(device), "grid": GRID, "width": args.width,
           "epochs": epochs, "n_train": n_tr, "n_test": n_te,
           "n_tasks": n_tasks, "Hs": Hs, "rhos": rhos,
           "single_step_hit": {h: single[h]["hit_rate"] for h in single},
           "params": {h: single[h]["params"] for h in single},
           "rows": rows}
    Path("p1_horizon_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 p1_horizon_results.json")


if __name__ == "__main__":
    main()
