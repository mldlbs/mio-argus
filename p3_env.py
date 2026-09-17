"""
P3 环境：带**进度信号**的多步 GUI 任务。

动机（见 P2_RESULTS.md）
------------------------
P2 证明：如果 verifier 必须"重新接地"来验证候选动作，
则 V(c) ≡ π(c)，验证与选择同构，H3 不成立。

H3 唯一还活着的形式是**结果验证**：
    V(旧屏幕, 动作, 新屏幕) → 状态是否推进？
这要求环境暴露**可观察的进度信号**。

环境设计
--------
- 屏幕顶部有**进度条**，只有正确动作才推进。
- 错误动作会在被点击的格子上留下**红色标记**，但进度条不动。
- 因此 "屏幕变了" ≠ "进度推进了"：
  两种动作都会改变屏幕，只有进度条能区分。
- 进度条是一个小而固定的区域 → 结果验证器可以做得**远小于策略**。

与 P0 环境的区别：多了进度条与错误标记，因此 π 需要重新训练（图像分布变了）。
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from p0_env import (EnvCfg, Element, PALETTE, LABEL_TO_ID, LABELS,
                    NUM_LABELS, _font)

BG = (246, 247, 249)
BAR_BG = (222, 226, 232)
BAR_FG = (52, 138, 72)
WRONG = (214, 60, 48)
DONE = (170, 176, 184)

BAR_Y0, BAR_Y1 = 2, 11


def _slot_box(slot: int, cfg: EnvCfg, top: int):
    """在 [top, size) 区域内划分网格，避免进度条与第 0 行元素重叠。"""
    size = cfg.img_size
    cell = (size - top) // cfg.grid
    r, c = divmod(slot, cfg.grid)
    pad = cfg.pad
    x1 = c * cell + pad
    y1 = top + r * cell + pad
    x2 = (c + 1) * cell - pad
    y2 = top + (r + 1) * cell - pad
    return x1, y1, x2, y2


def render_ui(labels: List[str], progress: int, H: int,
              wrong_mask: List[bool], done_mask: List[bool],
              cfg: EnvCfg, geom_seed: int = 0) -> Tuple[np.ndarray, List[Element]]:
    """
    渲染一帧。返回 (CHW float32, elements)。

    **几何与状态分离**：元素 bbox 的 jitter 只由 `geom_seed` 决定，
    与 progress/wrong/done 无关。因此同一布局在不同状态下渲染时，
    元素位置**逐像素一致**，帧间差异只来自进度条与标记
    （否则 jitter 会淹没真正的状态信号——P1 已踩过同样的坑）。
    """
    rng = np.random.default_rng(geom_seed)

    size = cfg.img_size
    top = BAR_Y1 + 4
    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)

    # 网格
    for slot in range(cfg.num_slots):
        x1, y1, x2, y2 = _slot_box(slot, cfg, top)
        d.rectangle([x1, y1, x2, y2], outline=(214, 218, 224))

    # 元素
    els = []
    for slot in range(cfg.num_slots):
        bx1, by1, bx2, by2 = _slot_box(slot, cfg, top)
        cw, ch = bx2 - bx1, by2 - by1
        ew = max(6, int(cw * cfg.fill) - int(rng.integers(0, 5)))
        eh = max(6, int(ch * cfg.fill) - int(rng.integers(0, 5)))
        ex1 = int(np.clip(bx1 + rng.integers(0, cfg.jitter + 1), 0, size - 2))
        ey1 = int(np.clip(by1 + rng.integers(0, cfg.jitter + 1), 0, size - 2))
        ex2 = int(np.clip(ex1 + ew, ex1 + 1, size - 1))
        ey2 = int(np.clip(ey1 + eh, ey1 + 1, size - 1))

        label = labels[slot]
        color = PALETTE[LABEL_TO_ID[label] % len(PALETTE)]
        if done_mask[slot]:
            color = DONE
        outline = WRONG if wrong_mask[slot] else (25, 25, 25)
        width = 3 if wrong_mask[slot] else 1

        d.rectangle([ex1, ey1, ex2, ey2], fill=color, outline=outline, width=width)
        if (ex2 - ex1) >= 30 and (ey2 - ey1) >= 14:
            f = _font(10)
            bb = d.textbbox((0, 0), label, font=f)
            d.text((ex1 + max(0, ((ex2 - ex1) - (bb[2] - bb[0])) // 2), ey1 + 2),
                   label, fill=(255, 255, 255), font=f)
        els.append(Element(slot, (ex1, ey1, ex2, ey2), label, "button", color))

    # 进度条（独立区域，与网格不重叠）
    x0, x1 = 3, size - 4
    d.rectangle([x0, BAR_Y0, x1, BAR_Y1], fill=BAR_BG, outline=(200, 204, 210))
    frac = progress / max(1, H)
    if frac > 0:
        xf = int(x0 + (x1 - x0) * frac)
        d.rectangle([x0, BAR_Y0 + 1, max(x0 + 1, xf), BAR_Y1 - 1], fill=BAR_FG)

    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1)), els


@dataclass
class StepState:
    labels: List[str]
    target: str
    progress: int
    wrong: List[bool]
    done: List[bool]


def fresh_labels(cfg: EnvCfg, rng) -> List[str]:
    vocab = LABELS[:cfg.label_vocab]
    idx = rng.choice(len(vocab), size=cfg.num_slots, replace=False)
    return [vocab[int(i)] for i in idx]


def make_episode(cfg: EnvCfg, rng, H: int, rho: float):
    """
    返回 [StepState, ...]，长度 H。
    每步布局以概率 ρ 重随机（与 P1 一致，用于控制"环境变化率"）。
    """
    labels = fresh_labels(cfg, rng)
    steps = []
    for t in range(H):
        if t > 0 and rng.random() < rho:
            labels = fresh_labels(cfg, rng)
        target = labels[int(rng.integers(0, cfg.num_slots))]
        steps.append(StepState(list(labels), target, t, [False] * cfg.num_slots,
                               [False] * cfg.num_slots))
    return steps


if __name__ == "__main__":
    cfg = EnvCfg(grid=6, img_size=160, label_vocab=36, fill=0.40, jitter=4)
    rng = np.random.default_rng(0)
    eps = make_episode(cfg, rng, H=4, rho=0.3)
    print("steps:", len(eps), "targets:", [s.target for s in eps])
    img, els = render_ui(eps[0].labels, 0, 4, eps[0].wrong, eps[0].done, cfg, rng)
    print("image:", img.shape)
    Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8)).save("_p3_preview.png")

    # 进度条差异演示
    img2, _ = render_ui(eps[0].labels, 2, 4, eps[0].wrong, eps[0].done, cfg, rng)
    diff = np.abs(img2 - img).max(axis=0)
    ys = np.where(diff.max(axis=1) > 0.1)[0]
    print("diff rows range:", ys.min(), ys.max(), "| bar rows:", BAR_Y0, BAR_Y1)
