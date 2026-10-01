"""
P0.5 廉价 UI 解析器：只用经典 CV（阈值 + 连通域），**不含任何学习模型**。

这是 H2 最脆弱假设的直接检验——P0 用的是 oracle 元素 bbox，
真实系统必须自己"看见"元素。这里给出一个诚实的廉价解析器，
并提供可调退化旋钮，用来扫描"解析器质量 → H2 优势"的关系。

解析步骤：
  1. 与背景色距离阈值 → 前景 mask
  2. 形态学开运算 → 去掉 1px 网格线
  3. 连通域 → 过滤面积 → bbox
"""
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

BG = (246, 247, 249)
GridLine = (214, 218, 224)


@dataclass
class ParseCfg:
    dist_thr: int = 30        # 与背景色的最大通道差
    min_area: int = 24        # 最小连通域面积（像素）
    max_area_frac: float = 0.25
    open_ksize: int = 3       # 形态学开运算核


def cheap_parse(img_u8: np.ndarray, cfg: ParseCfg = ParseCfg(),
                img_size: int = 160) -> List[Tuple[int, int, int, int]]:
    """img_u8: (H,W,3) uint8 → 候选 bbox 列表 (x1,y1,x2,y2)，像素坐标。"""
    bg = np.array(BG, dtype=np.int16)
    d = np.abs(img_u8.astype(np.int16) - bg).max(axis=2)
    mask = (d > cfg.dist_thr).astype(np.uint8)

    if cfg.open_ksize > 1:
        k = np.ones((cfg.open_ksize, cfg.open_ksize), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_area = cfg.max_area_frac * img_size * img_size

    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < cfg.min_area or area > max_area:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


# ---------- 可控退化 ----------

def degrade(boxes, rng: np.random.Generator, jit: int = 0, drop: float = 0.0,
            spurious: float = 0.0, shrink: float = 0.0,
            img_size: int = 160):
    """
    jit:       bbox 四边随机偏移 ±jit 像素
    drop:      每个 bbox 被丢弃的概率
    spurious:  每张图额外加入的假 bbox 数量（Poisson 均值）
    shrink:    bbox 尺寸收缩比例 [0,1)
    """
    out = []
    for (x1, y1, x2, y2) in boxes:
        if rng.random() < drop:
            continue
        if shrink > 0:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hw, hh = (x2 - x1) / 2 * (1 - shrink), (y2 - y1) / 2 * (1 - shrink)
            x1, x2 = cx - hw, cx + hw
            y1, y2 = cy - hh, cy + hh
        if jit > 0:
            x1 += rng.integers(-jit, jit + 1); x2 += rng.integers(-jit, jit + 1)
            y1 += rng.integers(-jit, jit + 1); y2 += rng.integers(-jit, jit + 1)
        x1 = int(np.clip(x1, 0, img_size - 2)); y1 = int(np.clip(y1, 0, img_size - 2))
        x2 = int(np.clip(x2, x1 + 1, img_size - 1)); y2 = int(np.clip(y2, y1 + 1, img_size - 1))
        out.append((x1, y1, x2, y2))

    k = rng.poisson(spurious)
    for _ in range(int(k)):
        x1 = int(rng.integers(0, img_size - 10)); y1 = int(rng.integers(0, img_size - 10))
        w = int(rng.integers(6, 30)); h = int(rng.integers(6, 30))
        out.append((x1, y1, min(img_size - 1, x1 + w), min(img_size - 1, y1 + h)))
    return out


# ---------- 质量指标 ----------

def center_in(box, target_bbox) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    tx1, ty1, tx2, ty2 = target_bbox
    return (tx1 <= cx <= tx2) and (ty1 <= cy <= ty2)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_index(dets, target_bbox):
    """返回覆盖目标中心的检测框下标（优先 IoU 最大），无则 -1。"""
    cand = [i for i, b in enumerate(dets) if center_in(b, target_bbox)]
    if not cand:
        return -1
    return max(cand, key=lambda i: iou(dets[i], target_bbox))


def quality(dets, target_bbox) -> dict:
    mi = match_index(dets, target_bbox)
    return {
        "covered": mi >= 0,
        "best_iou": max((iou(b, target_bbox) for b in dets), default=0.0),
        "n_det": len(dets),
        "match_idx": mi,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from p0_env import EnvCfg, make_sample

    cfg = EnvCfg(grid=6, img_size=160, label_vocab=36, fill=0.40, jitter=4)
    rng = np.random.default_rng(0)

    for name, kw in [("clean", {}),
                     ("jit2", {"jit": 2}),
                     ("drop30", {"drop": 0.3}),
                     ("jit3+drop30+spur2", {"jit": 3, "drop": 0.3, "spurious": 2.0})]:
        cov, ious, nd = [], [], []
        prng = np.random.default_rng(1)
        for _ in range(200):
            s = make_sample(rng, cfg)
            img = (s.image.transpose(1, 2, 0) * 255).astype(np.uint8)
            dets = degrade(cheap_parse(img, img_size=cfg.img_size), prng,
                           img_size=cfg.img_size, **kw)
            q = quality(dets, s.target_bbox)
            cov.append(q["covered"]); ious.append(q["best_iou"]); nd.append(q["n_det"])
        print(f"{name:<22} recall={np.mean(cov):.3f} "
              f"meanIoU={np.mean(ious):.3f} n_det={np.mean(nd):.1f}")
