"""
P0 环境：可控合成 GUI，同时暴露「像素」与「元素结构」两种接口。

设计要点（对应 HYPOTHESIS.md 第 6 节"环境是生死线"）：
- 难度可参数化（格子数、元素大小、颜色可辨识度、抖动）
- 确定性（种子）
- 结构已知（每个元素有 slot / bbox / label / role）
- 两种等价动作接口，判据统一为"执行点是否落在目标 bbox 内"：
    * 结构化  action = slot ∈ {0..S-1}   执行点 = 该 slot 元素中心
    * 像素    action = (x, y) ∈ [0,1]²
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABELS = [
    "save", "open", "send", "edit", "copy", "paste",
    "search", "home", "back", "next", "help", "close",
    "add", "remove", "share", "print", "undo", "redo",
    "zoom", "lock", "play", "pause", "stop", "reset",
    "up", "down", "left", "right", "ok", "cancel",
    "yes", "no", "new", "delete", "rename", "move",
    "run", "build", "test", "deploy", "sync", "pull",
    "push", "merge", "fork", "star", "watch", "issue",
    "docs", "about", "settings", "profile", "logout", "login",
    "import", "export", "refresh", "filter", "sort", "group",
    "pin", "flag", "mute", "volume",
]
LABEL_TO_ID = {v: i for i, v in enumerate(LABELS)}
NUM_LABELS = len(LABELS)


def make_palette(k: int) -> List[Tuple[int, int, int]]:
    """贪心最远点采样，在 RGB 立方体内取 k 个互相最远的颜色。"""
    k = min(k, 216)
    grid = [40, 96, 152, 208]
    candidates = [(r, g, b) for r in grid for g in grid for b in grid]
    rng = np.random.default_rng(0)
    idx = int(rng.integers(0, len(candidates)))
    picked = [candidates[idx]]
    dist = np.full(len(candidates), 1e18)
    for _ in range(k - 1):
        p = np.array(picked[-1], dtype=float)
        c = np.array(candidates, dtype=float)
        d = ((c - p) ** 2).sum(1)
        dist = np.minimum(dist, d)
        nxt = int(np.argmax(dist))
        if dist[nxt] <= 0:
            break
        picked.append(candidates[nxt])
    return picked


PALETTE = make_palette(64)


@dataclass
class EnvCfg:
    grid: int = 3              # 格子数 grid×grid = slot 数
    img_size: int = 128
    label_vocab: int = 12
    fill: float = 0.80         # 元素尺寸 / 格子尺寸（越小 → 精度要求越高）
    jitter: int = 6            # 元素在格子内的随机偏移
    color_by_label: bool = True
    pad: int = 3

    @property
    def num_slots(self) -> int:
        return self.grid * self.grid

    def slot_bbox(self, slot: int) -> Tuple[int, int, int, int]:
        r, c = divmod(slot, self.grid)
        cell = self.img_size // self.grid
        x1 = c * cell + self.pad
        y1 = r * cell + self.pad
        x2 = (c + 1) * cell - self.pad
        y2 = (r + 1) * cell - self.pad
        return (x1, y1, x2, y2)


DEFAULT = EnvCfg()


@dataclass
class Element:
    slot: int
    bbox: Tuple[int, int, int, int]
    label: str
    role: str
    color: Tuple[int, int, int]

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class Sample:
    image: np.ndarray
    instruction: str
    target_label: str
    target_slot: int
    target_xy: Tuple[float, float]
    target_bbox: Tuple[int, int, int, int]
    elements: List[Element] = field(default_factory=list)


def _font(size=11):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render(elements: List[Element], cfg: EnvCfg) -> np.ndarray:
    size = cfg.img_size
    img = Image.new("RGB", (size, size), (246, 247, 249))
    d = ImageDraw.Draw(img)
    font = _font(10)

    for slot in range(cfg.num_slots):
        x1, y1, x2, y2 = cfg.slot_bbox(slot)
        d.rectangle([x1, y1, x2, y2], outline=(214, 218, 224))

    for e in elements:
        x1, y1, x2, y2 = e.bbox
        d.rectangle([x1, y1, x2, y2], fill=e.color, outline=(25, 25, 25))
        if (x2 - x1) >= 30 and (y2 - y1) >= 14:
            bb = d.textbbox((0, 0), e.label, font=font)
            d.text((x1 + max(0, ((x2 - x1) - (bb[2] - bb[0])) // 2), y1 + 2),
                   e.label, fill=(255, 255, 255), font=font)

    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def make_sample(rng: np.random.Generator, cfg: EnvCfg = DEFAULT) -> Sample:
    vocab = LABELS[:cfg.label_vocab]
    # 唯一性优先：能达到则不放回，否则放回
    replace = cfg.num_slots > len(vocab)
    assigned = rng.choice(len(vocab), size=cfg.num_slots, replace=replace)

    elements: List[Element] = []
    for slot in range(cfg.num_slots):
        bx1, by1, bx2, by2 = cfg.slot_bbox(slot)
        cw, ch = bx2 - bx1, by2 - by1
        ew = max(6, int(cw * cfg.fill) - int(rng.integers(0, 5)))
        eh = max(6, int(ch * cfg.fill) - int(rng.integers(0, 5)))
        ex1 = int(np.clip(bx1 + rng.integers(0, cfg.jitter + 1), 0, cfg.img_size - 2))
        ey1 = int(np.clip(by1 + rng.integers(0, cfg.jitter + 1), 0, cfg.img_size - 2))
        ex2 = int(np.clip(ex1 + ew, ex1 + 1, cfg.img_size - 1))
        ey2 = int(np.clip(ey1 + eh, ey1 + 1, cfg.img_size - 1))

        label = vocab[int(assigned[slot])]
        color = PALETTE[LABEL_TO_ID[label] % len(PALETTE)] if cfg.color_by_label \
            else PALETTE[int(rng.integers(0, len(PALETTE)))]

        elements.append(Element(slot, (ex1, ey1, ex2, ey2), label,
                                str(rng.choice(["button", "input", "icon"])), color))

    target_slot = int(rng.integers(0, cfg.num_slots))
    t = elements[target_slot]
    return Sample(
        image=render(elements, cfg),
        instruction=f"click {t.label}",
        target_label=t.label,
        target_slot=target_slot,
        target_xy=(t.center[0] / cfg.img_size, t.center[1] / cfg.img_size),
        target_bbox=t.bbox,
        elements=elements,
    )


def hit(pred_xy: Tuple[float, float], bbox: Tuple[int, int, int, int],
        cfg: EnvCfg = DEFAULT) -> bool:
    x, y = pred_xy[0] * cfg.img_size, pred_xy[1] * cfg.img_size
    x1, y1, x2, y2 = bbox
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def cell_of(xy: Tuple[float, float], cfg: EnvCfg = DEFAULT) -> int:
    g = cfg.grid
    cx = int(np.clip(xy[0] * g, 0, g - 1))
    cy = int(np.clip(xy[1] * g, 0, g - 1))
    return cy * g + cx


def instruction_onehot(instruction: str, n: int = NUM_LABELS) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    w = instruction.split()[-1]
    if w in LABEL_TO_ID and LABEL_TO_ID[w] < n:
        v[LABEL_TO_ID[w]] = 1.0
    return v


if __name__ == "__main__":
    for g in (3, 5, 8):
        cfg = EnvCfg(grid=g, fill=0.65)
        rng = np.random.default_rng(0)
        s = make_sample(rng, cfg)
        print(f"grid={g} slots={cfg.num_slots} img={s.image.shape} "
              f"tgt_slot={s.target_slot} instr='{s.instruction}'")
    cfg = EnvCfg(grid=5, fill=0.65, img_size=160)
    rng = np.random.default_rng(1)
    s = make_sample(rng, cfg)
    Image.fromarray((s.image.transpose(1, 2, 0) * 255).astype(np.uint8)).save("_p0_preview.png")
    print("wrote _p0_preview.png")
