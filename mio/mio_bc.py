"""
MioGUI 监督基线（序列模仿第一步）

目的：证明 mio_env 可被监督训练（"先监督/序列模仿，再 RL"的第一档）。
极简：结构化接口 + 指令匹配 + stale→reobserve。

关键实现决定：**两处解耦**
  - 选哪个 slot：归一化嵌入的点积匹配（softmax over slots）
  - 是否 reobserve：独立 sigmoid 门（输入 stale）
  早期版本把 reobserve 塞进同一个 softmax，训练塌缩（train acc 卡 0.23）。

输入：obs["elements"] 的 slot 标签、指令标签、stale 标志
输出：动作分类（0..S-1 = click slot，S = reobserve）

用法:
    python mio/mio_bc.py --quick
    python mio/mio_bc.py
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in (str(_ROOT / "experiments" / "paper"), str(_ROOT)) if p not in _sys.path]

from p0_env import LABEL_TO_ID
from mio_env import MioCfg, MioGUIEnv, sample_expert_dataset

NLAB = len(LABEL_TO_ID)


def featurize(records, num_slots):
    S = num_slots
    sl = np.zeros((len(records), S), dtype=np.int64)
    instr = np.zeros(len(records), dtype=np.int64)
    stale = np.zeros(len(records), dtype=np.float32)
    tgt = np.zeros(len(records), dtype=np.int64)
    for i, r in enumerate(records):
        for e in r["obs"]["elements"]:
            sl[i, e.slot] = LABEL_TO_ID[e.label]
        instr[i] = LABEL_TO_ID[r["obs"]["instruction"].split()[-1]]
        stale[i] = float(r["obs"]["stale"])
        a = r["action"]
        tgt[i] = a["slot"] if a["type"] == "click" else S
    return sl, instr, stale, tgt


def to_tensors(sl, instr, stale, tgt, device):
    return (torch.tensor(sl, device=device), torch.tensor(instr, device=device),
            torch.tensor(stale, device=device), torch.tensor(tgt, device=device))


class BCPolicy(nn.Module):
    def __init__(self, d=64, scale=8.0):
        super().__init__()
        self.es = nn.Embedding(NLAB, d)     # slot 标签嵌入
        self.ei = nn.Embedding(NLAB, d)     # 指令标签嵌入
        self.scale = scale
        self.gate = nn.Sequential(nn.Linear(1, d), nn.ReLU(),
                                  nn.Linear(d, 1))

    def forward(self, slot_labels, instr, stale):
        key = F.normalize(self.es(slot_labels), dim=-1)        # (B,S,d)
        q = F.normalize(self.ei(instr), dim=-1)                # (B,d)
        slot_logits = self.scale * (key * q[:, None, :]).sum(-1)   # (B,S)
        # 门只吃 stale：与匹配头解耦，避免梯度污染 ei（早期版本共享 ei → 匹配塌缩）
        g = self.gate(stale[:, None]).squeeze(-1)              # (B,)
        return slot_logits, g

    def act(self, obs, device):
        sl = np.zeros((1, env_slots(obs)), dtype=np.int64)
        for e in obs["elements"]:
            sl[0, e.slot] = LABEL_TO_ID[e.label]
        instr = np.array([LABEL_TO_ID[obs["instruction"].split()[-1]]])
        stale = np.array([float(obs["stale"])], dtype=np.float32)
        with torch.no_grad():
            slot_logits, g = self.forward(
                *to_tensors(sl, instr, stale, np.zeros(1, dtype=np.int64),
                            device)[:3])
        if float(torch.sigmoid(g[0])) > 0.5:
            return {"type": "reobserve"}, True
        return {"type": "click", "slot": int(slot_logits[0].argmax())}, False


def env_slots(obs):
    return max(e.slot for e in obs["elements"]) + 1


def rollout(env, model, device, episodes, seed=5000):
    succ, rets, reobs = [], [], []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        R, nro, guard = 0.0, 0, 0
        while not env.done and guard < 500:
            guard += 1
            action, is_reob = model.act(obs, device)
            nro += int(is_reob)
            obs, r, done, info = env.step(action)
            R += r
        succ.append(env.t >= env.cfg.H)
        rets.append(R)
        reobs.append(nro)
    return float(np.mean(succ)), float(np.mean(rets)), float(np.mean(reobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="默认 stale")
    args = ap.parse_args()

    cfg = MioCfg(grid=6, label_vocab=48, H=5, rho=0.3,
                 observe_mode="fresh" if args.fresh else "stale")
    n_ep = args.episodes if args.episodes else (200 if args.quick else 600)
    epochs = args.epochs if args.epochs else (40 if args.quick else 100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MioGUIEnv(cfg)
    print(f"observe={cfg.observe_mode} slots={env.num_slots} H={cfg.H} "
          f"rho={cfg.rho} n_ep={n_ep} epochs={epochs} device={device}")

    recs, _ = sample_expert_dataset(n_ep, cfg, seed=1000)
    sl, instr, stale, tgt = featurize(recs, env.num_slots)
    is_reob = (tgt == env.num_slots)
    print(f"dataset: {len(recs)} steps; reobserve 占比={float(is_reob.mean()):.3f}")
    tsl, ti, ts, tt = to_tensors(sl, instr, stale, tgt, device)
    ire = torch.tensor(is_reob, device=device)

    model = BCPolicy().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    B = 256
    for ep in range(epochs):
        perm = torch.randperm(len(recs))
        for i in range(0, len(recs), B):
            idx = perm[i:i + B]
            slot_logits, g = model(tsl[idx], ti[idx], ts[idx])
            not_reob = ~ire[idx]
            loss = torch.zeros((), device=device)
            if not_reob.any():
                loss = loss + F.cross_entropy(slot_logits[not_reob], tt[idx][not_reob])
            loss = loss + F.binary_cross_entropy_with_logits(g, ire[idx].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  epoch {ep+1}: loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        slot_logits, g = model(tsl, ti, ts)
        pred = slot_logits.argmax(-1)
        gate_pred = (torch.sigmoid(g) > 0.5)
        # 正确 = (该 reobserve 且门开) 或 (该点击且门关且 slot 对)
        correct = torch.where(ire, gate_pred,
                              (~gate_pred) & (pred == tt))
        acc = correct.float().mean().item()
        slot_acc = (pred[~ire] == tt[~ire]).float().mean().item() if (~ire).any() else float("nan")
        gate_acc = (gate_pred == ire).float().mean().item()
    print(f"\ntrain: 总acc={acc:.3f} slot匹配acc={slot_acc:.3f} gate acc={gate_acc:.3f}")

    s, r, ro = rollout(env, model, device, 100)
    print(f"BC rollout(100ep): success={s:.3f} return={r:.3f} reobserves/ep={ro:.2f}")
    print("（对照 expert：success=1.0；stale 下 reobserves≈1.1）")


if __name__ == "__main__":
    main()
