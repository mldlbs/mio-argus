"""
MioGUI 训练环境：gym 风格的 episodic 交互环境（任务环境，非探针）

动机
----
Q1–Q3 都是"探针"，无法训练可塑/元认知/世界模型类组件。本文件提供真正可训练的
环境：`reset()/step()`、ground-truth 奖励、可非平稳、可部分观测。

任务
----
多步 GUI grounding（复用 p0_env 的网格渲染）：
  每个 step 给一条指令 "click <label>"，agent 需点击**真实当前布局**中该 label 的元素。
  点对 → 进度 +1；点错 → 不推进。H 步后 episode 结束。
  每步之后布局以概率 ρ 重随机（长程 + 错误恢复）。
  可选 rule_permute：每个 episode 重排 label→color（跨 episode 非平稳，练可塑）。

为训练而设计的四个开关
----------------------
1. `observe_mode`：fresh（每步自动刷新）/ stale（只在 reset/reobserve 时刷新）。
   stale 让"重观察"成为有代价的决策——这正是元认知需要的成本模型。
2. `reobserve_cost`：reobserve 的即时代价（写成负 reward）。
3. `rule_permute`：episode 级非平稳。
4. 双接口：obs 同时给 `pixels` 与 `elements`，可分别练像素策略与结构策略（H2）。

接口
----
    env = MioGUIEnv(cfg)
    obs = env.reset(seed=...)
    obs, reward, done, info = env.step({"type": "click", "slot": 3})
    env.step({"type": "click_xy", "x": 0.4, "y": 0.6})
    env.step({"type": "reobserve"})
    env.expert_action() -> dict          # oracle（用于模仿）
    sample_expert_dataset(...)           # 供监督/序列模仿

用法:
    python mio/mio_env.py --demo
    python mio/mio_env.py --demo --stale --permute
"""
import argparse
from dataclasses import dataclass

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in (str(_ROOT / "experiments" / "paper"), str(_ROOT)) if p not in _sys.path]

from p0_env import (EnvCfg, Element, render,
                    PALETTE, LABEL_TO_ID, LABELS)


@dataclass
class MioCfg:
    grid: int = 6
    img_size: int = 160
    label_vocab: int = 32
    fill: float = 0.72
    jitter: int = 5
    pad: int = 3
    H: int = 5                  # 每 episode 步数
    rho: float = 0.3            # 每步之后布局重随机概率
    rule_permute: bool = False  # 每 episode 重排 label→color
    observe_mode: str = "fresh"  # fresh | stale
    reobserve_cost: float = 0.05
    correct_reward: float = 1.0
    wrong_penalty: float = 0.0
    max_attempts: int = 8       # 每步最多尝试次数（防止死循环）


class MioGUIEnv:
    def __init__(self, cfg: MioCfg = None):
        self.cfg = cfg or MioCfg()
        c = self.cfg
        self.ecfg = EnvCfg(grid=c.grid, img_size=c.img_size,
                           label_vocab=c.label_vocab, fill=c.fill,
                           jitter=c.jitter, color_by_label=False, pad=c.pad)
        self.num_slots = self.ecfg.num_slots
        self.vocab = LABELS[:c.label_vocab]
        self.rng = np.random.default_rng(0)
        self.t = 0
        self.done = True
        self.layout = None
        self.obs_layout = None
        self.target = None
        self.attempts = 0
        self.episode = 0

    # ---------------- 内部：布局 ----------------

    def _new_perm(self):
        p = np.arange(len(PALETTE))
        self.rng.shuffle(p)
        return p

    def _make_layout(self):
        c = self.cfg
        replace = self.num_slots > len(self.vocab)
        assigned = self.rng.choice(len(self.vocab), size=self.num_slots,
                                   replace=replace)
        els = []
        for slot in range(self.num_slots):
            bx1, by1, bx2, by2 = self.ecfg.slot_bbox(slot)
            cw, ch = bx2 - bx1, by2 - by1
            ew = max(6, int(cw * c.fill) - int(self.rng.integers(0, 5)))
            eh = max(6, int(ch * c.fill) - int(self.rng.integers(0, 5)))
            ex1 = int(np.clip(bx1 + self.rng.integers(0, c.jitter + 1), 0,
                              c.img_size - 2))
            ey1 = int(np.clip(by1 + self.rng.integers(0, c.jitter + 1), 0,
                              c.img_size - 2))
            ex2 = int(np.clip(ex1 + ew, ex1 + 1, c.img_size - 1))
            ey2 = int(np.clip(ey1 + eh, ey1 + 1, c.img_size - 1))
            label = self.vocab[int(assigned[slot])]
            idx = LABEL_TO_ID[label] % len(PALETTE)
            color = tuple(int(v) for v in PALETTE[self.perm[idx]])
            els.append(Element(slot, (ex1, ey1, ex2, ey2), label,
                               "button", color))
        return els

    def _sample_target(self):
        labels = [e.label for e in self.layout]
        return labels[int(self.rng.integers(0, len(labels)))]

    # ---------------- API ----------------

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.perm = self._new_perm()
        self.layout = self._make_layout()
        self.obs_layout = self.layout
        self.t = 0
        self.attempts = 0
        self.done = False
        self.episode += 1
        self.target = self._sample_target()
        return self._obs()

    def _obs(self):
        img = render(self.obs_layout, self.ecfg)
        return {
            "pixels": img,                       # (3,S,S) float32 [0,1]
            "elements": list(self.obs_layout),   # 结构接口
            "instruction": f"click {self.target}",
            "step": self.t,
            "H": self.cfg.H,
            "stale": self.obs_layout is not self.layout,
        }

    def _click(self, slot):
        el = next((e for e in self.layout if e.slot == slot), None)
        return el is not None and el.label == self.target

    def step(self, action):
        assert not self.done, "episode 已结束，请 reset()"
        c = self.cfg
        r, info = 0.0, {"reobserve": False, "correct": False}
        at = action["type"]
        self.attempts += 1          # 任何动作都计入预算（否则 reobserve 会 livelock）

        if at == "reobserve":
            if c.observe_mode == "stale":
                self.obs_layout = self.layout
            r -= c.reobserve_cost
            info["reobserve"] = True
        elif at in ("click", "click_xy"):
            if at == "click":
                slot = int(action["slot"])
            else:
                x = action["x"] * c.img_size
                y = action["y"] * c.img_size
                hit_el = next((e for e in self.layout
                               if e.bbox[0] <= x <= e.bbox[2]
                               and e.bbox[1] <= y <= e.bbox[3]), None)
                slot = hit_el.slot if hit_el else -1
            if self._click(slot):
                r += c.correct_reward
                info["correct"] = True
                self.t += 1
                self.attempts = 0
                if self.t >= c.H:
                    self.done = True
                else:
                    if self.rng.random() < c.rho:
                        self.layout = self._make_layout()
                        if c.observe_mode == "fresh":
                            self.obs_layout = self.layout
                    self.target = self._sample_target()
            else:
                r += c.wrong_penalty
        else:
            raise ValueError(f"未知动作: {action}")

        if not self.done and self.attempts >= c.max_attempts:
            self.done = True
            info["stuck"] = True

        if c.observe_mode == "fresh" and not self.done:
            self.obs_layout = self.layout
        return self._obs(), float(r), self.done, info

    # ---------------- oracle（供模仿/监督） ----------------

    def expert_action(self):
        """教师：若观察已过期则重观察，否则点击目标 slot。"""
        if self.obs_layout is not self.layout:
            return {"type": "reobserve"}
        el = next((e for e in self.layout if e.label == self.target), None)
        if el is None:
            return {"type": "reobserve"}
        return {"type": "click", "slot": el.slot}


# ---------------- 轨迹采样（监督/序列模仿用） ----------------

def sample_expert_dataset(n_episodes, cfg=None, seed=0, max_steps=None):
    """返回逐 step 的 (obs, action, reward, done)。"""
    cfg = cfg or MioCfg()
    env = MioGUIEnv(cfg)
    records = []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        steps = 0
        while not env.done and (max_steps is None or steps < max_steps):
            a = env.expert_action()
            pre_obs = obs                      # 与动作配对的必须是**动作前**的观察
            obs, r, done, info = env.step(a)
            records.append({"obs": pre_obs, "action": a, "reward": r,
                            "done": done, "episode": ep})
            steps += 1
    return records, env


def expert_action_labels(records, cfg):
    """把动作转成分类标签：0..num_slots-1 = click(slot)，num_slots = reobserve。"""
    return np.array([r["action"]["slot"] if r["action"]["type"] == "click"
                     else cfg.grid * cfg.grid for r in records], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--stale", action="store_true")
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    cfg = MioCfg(observe_mode="stale" if args.stale else "fresh",
                 rule_permute=args.permute)
    env = MioGUIEnv(cfg)
    print(f"MioGUIEnv: grid={cfg.grid} slots={env.num_slots} H={cfg.H} "
          f"rho={cfg.rho} observe={cfg.observe_mode} permute={cfg.rule_permute}")

    rets, succ, n_reobs, ep_lens = [], [], [], []
    for ep in range(args.episodes):
        obs = env.reset(seed=1000 + ep)
        R, nro = 0.0, 0
        while not env.done:
            a = env.expert_action()
            if a["type"] == "reobserve":
                nro += 1
            obs, r, done, info = env.step(a)
            R += r
        rets.append(R)
        succ.append(env.t >= cfg.H)
        n_reobs.append(nro)
        ep_lens.append(env.t)
        if ep < 3:
            print(f"  ep{ep}: steps_ok={env.t}/{cfg.H} reobserves={nro} "
                  f"return={R:.3f}")
    print(f"\nexpert over {args.episodes} episodes: "
          f"success={np.mean(succ):.3f} return={np.mean(rets):.3f} "
          f"reobserves/ep={np.mean(n_reobs):.2f} steps_ok={np.mean(ep_lens):.2f}")

    recs, _ = sample_expert_dataset(10, cfg, seed=1000)
    print(f"sample_expert_dataset: {len(recs)} step records; "
          f"obs keys={list(recs[0]['obs'].keys())}")


if __name__ == "__main__":
    main()
