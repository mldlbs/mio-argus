"""
P6：噪声验证器 —— "验证器要多准" 随 horizon 如何变化？

动机（REPORT.md §10 声明的局限）
--------------------------------
P3 的进度信号是**合成、无噪声**的，V 达到 acc=1.0。
真实环境里 verifier 必然有错：进度条会误报、延迟、被遮挡。

本实验给 V 注入**可控错误率**，回答：

  Q: 达到同等收益所需的验证器准确率，是否随 horizon H 下降？

预测：**是**。因为每步的验证错误会复利，H 越大容错越低。
若预测成立 → 给出一个可操作的结论：
    "H=8 的任务要求 verifier 错误率 < X%"，X 可由实验测出。

设计
----
三种错误模式（区分开，因为两种错误的代价不同）：

  fnr  仅假阴性：正确动作被 V 误拒  → 丢掉正确动作，该步失败
  fpr  仅假阳性：错误动作被 V 误收  → 以为推进了，实际没有
  both 对称翻转

扫描 ε ∈ {0, 0.02, 0.05, 0.1, 0.2, 0.4}，H ∈ {1,2,4,8}。

用法:
    python p6_noisy_verifier.py --quick
    python p6_noisy_verifier.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import p3_verifier as P3
from p3_env import make_episode, fresh_labels, render_ui

GRID = 6
S = GRID * GRID
EPS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]


def rollout_noisy(pi, V, cfg, device, H, rho, n_tasks, seed,
                  max_retry=3, v_err=0.0, err_mode="both", v_thr=0.5):
    """
    与 P3 的 rollout 相同，但 V 的判断以概率 v_err 出错。
    err_mode 决定注入哪种错误（需要知道动作是否正确才能区分）。
    """
    rng = np.random.default_rng(seed)
    # 噪声翻转必须用**独立** rng：否则不同 ε 会消耗不同数量的随机数，
    # 导致每次跑的任务都不一样，ε 之间不可比（踩过这个坑）。
    noise_rng = np.random.default_rng(seed + 999_999)
    task_ok = []

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

            tgt_slot = labels.index(steps[t].target)
            p = P3.pi_probs(pi, img, steps[t].target, device)
            order = list(np.argsort(-p))
            tried = set()

            for _ in range(max_retry):
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

                accept = P3.v_advanced(V, img, nimg, device, v_thr)
                if v_err > 0:
                    flip = noise_rng.random() < v_err
                    if flip:
                        if err_mode == "both":
                            accept = not accept
                        elif err_mode == "fnr" and correct and accept:
                            accept = False
                        elif err_mode == "fpr" and (not correct) and (not accept):
                            accept = True

                if accept:
                    wrong, done, progress, img = nw, nd, nprog, nimg
                    break
                else:
                    wrong, done, img = nw, nd, nimg

            if progress != t + 1:
                success = False
                break

        task_ok.append(success)

    return {"task_success": float(np.mean(task_ok))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--v-epochs", type=int, default=None)
    ap.add_argument("--tasks", type=int, default=None)
    args = ap.parse_args()

    epochs = args.epochs if args.epochs else (10 if args.quick else 25)
    v_ep = args.v_epochs if args.v_epochs else (8 if args.quick else 15)
    n_tr = 1500 if args.quick else 4000
    n_te = 500 if args.quick else 1000
    n_tasks = args.tasks if args.tasks else (80 if args.quick else 300)
    Hs = [1, 2, 4] if args.quick else [1, 2, 4, 8]
    eps_list = [0.0, 0.1, 0.4] if args.quick else EPS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = P3.cfg_p3()
    print(f"device={device} H={Hs} eps={eps_list} tasks={n_tasks}")
    print(f"错误模式: fnr / fpr / both，rho=1.0\n")

    # 训练 π 与 V（V 用校准版，修复已知的阈值漂移缺陷）
    ptr = P3.build_pi_data(n_tr, 0, cfg)
    pte = P3.build_pi_data(n_te, 999, cfg)
    pi = P3.train_pi(32, ptr, epochs, device)
    pa = P3.pi_acc(pi, pte, device)
    vtr = P3.build_v_data(n_tr, 1, cfg)
    vcal = P3.build_v_data(n_te, 555, cfg)      # 校准用验证集
    vte = P3.build_v_data(n_te, 998, cfg)       # 报告用测试集
    V, vrec = P3.train_v_calibrated(8, vtr, vcal, v_ep, device, seed=0)
    v_thr = vrec["thr"]
    vm = P3.v_eval(V, vte, device)
    sc = P3.v_scores_all(V, vte, device)
    y = vte["label"].astype(int)
    cal_acc = float(((sc > v_thr).astype(int) == y).mean())
    print(f"[π] single-step acc={pa:.4f}   "
          f"[V] params={V.n_params():,} auc={vrec['val_auc']:.4f} "
          f"thr={v_thr:.4f} test_acc@thr={cal_acc:.4f}")
    print(f"    (若用固定 0.5 阈值: acc={vm['acc']:.4f}, "
          f"tpr={vm['tpr']:.4f}, fpr={vm['fpr']:.4f})\n")

    rows = []
    for mode in ("fnr", "fpr", "both"):
        print(f"### 错误模式 = {mode}")
        print(f"{'eps':>6}" + "".join(f"{'H='+str(h):>10}" for h in Hs))
        print("-" * (6 + 10 * len(Hs)))
        base = {}
        for H in Hs:
            base[H] = rollout_noisy(pi, V, cfg, device, H, 1.0, n_tasks,
                                    3000 + H, 3, 0.0, mode, v_thr)["task_success"]
        for eps in eps_list:
            row = {"mode": mode, "eps": eps}
            cells = []
            for H in Hs:
                if eps == 0.0:
                    v = base[H]
                else:
                    v = rollout_noisy(pi, V, cfg, device, H, 1.0, n_tasks,
                                      3000 + H, 3, eps, mode, v_thr)["task_success"]
                row[f"H{H}"] = v
                cells.append(v)
            rows.append(row)
            print(f"{eps:>6.2f}" + "".join(f"{v:>10.4f}" for v in cells))
        print()

    # ---- 每个 H 的"崩溃点"：收益跌到 0 的 eps ----
    print(f"{'='*66}")
    print("判定：各 H 下收益归零的 ε（验证器可容忍错误率）")
    print(f"{'H':>4}{'eps=0 succ':>13}{'崩溃 ε':>12}{'半衰 ε':>12}")
    print("-" * 44)
    for H in Hs:
        b = next(r for r in rows if r["mode"] == "both" and r["eps"] == 0.0)[f"H{H}"]
        half = b / 2
        crash, half_eps = None, None
        for r in rows:
            if r["mode"] != "both" or r["eps"] == 0.0:
                continue
            if half_eps is None and r[f"H{H}"] <= half:
                half_eps = r["eps"]
            if r[f"H{H}"] <= 0.0:
                crash = r["eps"]
                break
        print(f"{H:>4}{b:>13.4f}"
              f"{(f'{crash:.2f}' if crash is not None else '>0.40'):>12}"
              f"{(f'{half_eps:.2f}' if half_eps is not None else '>0.40'):>12}")
    print("=" * 66)

    out = {"device": str(device), "grid": GRID, "Hs": Hs, "eps": eps_list,
           "n_tasks": n_tasks, "pi_acc": pa, "v_metrics": vm, "rows": rows}
    Path("p6_noisy_verifier_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写出 p6_noisy_verifier_results.json")


if __name__ == "__main__":
    main()
