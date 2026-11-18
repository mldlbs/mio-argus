"""H3b 泛化验证 runner（新文件，不改 P3）。

三个未见条件：
  C0 grid8  ：8×8 网格（64 槽位）
  C1 eps15  ：V 决策噪声 ε=0.15（超出 P6 的 0.1）
  C2 delay1 ：V 只能裁决"上一次尝试"（延迟 1 步）

两臂：pi（无 V）vs v（V 门控重试）。同一 (条件, seed) 下两臂共享同一批任务流。
判定与口径见 results/paper/H3B_GENERALIZATION_PREREG.md v0.2。
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import p3_verifier as P3
from p0_env import EnvCfg
from p0_env import LABELS
from p3_env import fresh_labels, make_episode, render_ui

IMG_SIZE = 160
HS = [1, 2, 4, 8]
N_TASKS = 300
RHO = 0.3
MAX_RETRY = 3


def cfg_for_cond(cond: str) -> EnvCfg:
    grid = 8 if cond == "grid8" else 6
    return EnvCfg(grid=grid, img_size=IMG_SIZE, label_vocab=grid * grid,
                  fill=0.40, jitter=4, color_by_label=True)


def train_pi_cfg(width, tr, epochs, device, cfg, seed=0, lr=3e-3, batch=64):
    """与 P3.train_pi 等价，但 grid 跟随 cfg（P3 里写死 GRID=6，8×8 会越界）。"""
    import torch.nn.functional as F
    from p0_models import P0Net
    from p0_h2 import to_t

    torch.manual_seed(seed)
    m = P0Net(width=width, head="cls", num_slots=cfg.num_slots, grid=cfg.grid,
              num_labels=len(LABELS), in_res=IMG_SIZE, stages=3).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    images = to_t(tr["images"], device)
    inst = to_t(tr["inst"], device)
    tgt = torch.from_numpy(tr["slots"]).to(device)
    n = images.shape[0]
    spe = max(1, n // batch)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(spe):
            idx = perm[s * batch:(s + 1) * batch]
            loss = F.cross_entropy(m(images[idx], inst[idx]), tgt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def decide(v, thr, before, after, eps, rng):
    """V 的判定（可加噪声翻转）。输入必须是**允许的信息**。"""
    acc = P3.v_advanced(v, before, after, DEV, thr) if v is not None else True
    if eps > 0 and rng.random() < eps:
        acc = not acc
    return bool(acc)


def rollout(pi, v, thr, cfg, device, H, rho, n_tasks, seed,
            max_retry=MAX_RETRY, mode="pi", eps=0.0, delay=0):
    rng = np.random.default_rng(seed)
    S = cfg.num_slots
    task_ok, retries = [], []
    for ti in range(n_tasks):
        steps = make_episode(cfg, rng, H, rho)
        labels = list(steps[0].labels)
        wrong = [False] * S
        done = [False] * S
        progress = 0
        geom = ti * 100
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
            p = P3.pi_probs(pi, img, target, device)
            order = list(np.argsort(-p))
            tried = set()
            pending = None          # (snapshot_before, img_after) — 延迟模式下待裁决
            used = 0

            R = 1 if mode == "pi" else max_retry
            for k in range(R):
                # 延迟 1 步：先裁决**上一次**尝试，再决定这次是否能从当前状态继续
                if delay > 0 and pending is not None:
                    if not decide(v, thr, pending[0][0], pending[1], eps, rng):
                        img, wrong, done, progress = pending[0]
                        pending = None
                        continue

                cand = [c for c in order if c not in tried]
                slot = int((cand or order)[0])
                tried.add(slot)
                used += 1

                snap = (img.copy(), list(wrong), list(done), progress)
                correct = (slot == tgt_slot)
                nw, nd = list(wrong), list(done)
                if correct:
                    nd[slot] = True
                else:
                    nw[slot] = True
                nprog = progress + (1 if correct else 0)
                nimg, _ = render_ui(labels, nprog, H, nw, nd, cfg, geom_seed=geom)

                img, wrong, done, progress = nimg, nw, nd, nprog
                if mode == "pi":
                    break
                if delay > 0:
                    pending = (snap, nimg)      # 暂存，下一次尝试前裁决
                    continue
                verdict = decide(v, thr, snap[0], nimg, eps, rng)
                if verdict:
                    break
                img, wrong, done, progress = snap

            # 延迟模式：循环结束时，最后一次尝试的裁决仍要用上
            if delay > 0 and pending is not None:
                if not decide(v, thr, pending[0][0], pending[1], eps, rng):
                    img, wrong, done, progress = pending[0]

            retries.append(used - 1)
            if progress != t + 1:
                success = False
                break
        task_ok.append(success)
    return {"task_success": float(np.mean(task_ok)),
            "mean_retries": float(np.mean(retries)) if retries else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=("grid8", "eps15", "delay1"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--v-epochs", type=int, default=15)
    ap.add_argument("--tasks", type=int, default=N_TASKS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/paper/h3b_runs.jsonl")
    args = ap.parse_args()

    global DEV
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = cfg_for_cond(args.cond)
    n_tr = 1500 if args.quick else 4000
    eps = 0.15 if args.cond == "eps15" else 0.0
    delay = 1 if args.cond == "delay1" else 0

    t0 = time.time()
    tr_pi = P3.build_pi_data(n_tr, 0, cfg)
    pi = train_pi_cfg(32, tr_pi, args.epochs, DEV, cfg, seed=args.seed)
    pi_acc = P3.pi_acc(pi, tr_pi, DEV)

    v_tr = P3.build_v_data(n_tr, 1, cfg)
    v_val = P3.build_v_data(400, 2, cfg)
    v, vrec = P3.train_v_calibrated(8, v_tr, v_val, args.v_epochs, DEV, seed=args.seed)
    thr = float(vrec["thr"])
    vm = P3.v_eval(v, v_val, DEV)

    row = {"cond": args.cond, "seed": args.seed, "pi_acc": round(float(pi_acc), 4),
           "v_acc": round(float(vm["acc"]), 4), "thr": round(float(thr), 4),
           "params_pi": pi.n_params(), "params_v": v.n_params(),
           "eps": eps, "delay": delay, "tasks": args.tasks, "H8": {}}

    for arm in ("pi", "v"):
        for H in HS:
            r = rollout(pi, v if arm == "v" else None, thr, cfg, DEV, H, RHO,
                        args.tasks, seed=5000 + 100 * ["grid8", "eps15", "delay1"].index(args.cond)
                        + H * 7 + int(RHO * 100),
                        mode=arm, eps=eps, delay=delay)
            row[f"{arm}_H{H}"] = round(r["task_success"], 4)
            if H == 8:
                row["H8"][arm] = {"succ": round(r["task_success"], 4),
                                  "retries": round(r["mean_retries"], 2)}
    row["delta_H8"] = round(row["v_H8"] - row["pi_H8"], 4)
    row["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with Path(args.out).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
