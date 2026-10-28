"""
W28b 微调：接环示范数据（DemoSample v2）→ LoRA 微调（A' 路线）

与 W30 的差别：
- 只训 LoRA + 三个头（ViT/GPT-2 冻结），从 v3 checkpoint 迁移学习；
- 坐标用 demo.norm_coord（屏无关，= x/W, y/H），不再按 1920x1080 硬编码，
  这样 2560x1440 采集的数据与 1920x1080 的旧数据在同一坐标系；
- 按任务（task_id）切分 train/val，避免同屏泄漏；
- 多应用 manifest：--apps 选训练应用（notepad / qgis / probe），用于 E1/E2/E3 对照。

坐标回归目标定义：click/move 用 norm_coord；其它动作 coord_mask=False。

用法:
    python experiments/computer_use/train_inloop.py --apps notepad
    python experiments/computer_use/train_inloop.py --apps qgis --epochs 30
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(ROOT / "mio"), str(ROOT)) if p not in sys.path]

from mio_argus import (  # noqa: E402
    ACTION_MAP,
    ACTION_TO_IDX,
    ComputerUseModelV3,
    get_image_processor,
    get_tokenizer,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_ROOT = ROOT / "artifacts" / "computer_use_inloop"
CKPT_DIR = ROOT / "artifacts" / "checkpoints"
BASE_CKPT = CKPT_DIR / "computer_use_model_v3_real.pth"
MAX_LEN = 64
BATCH_SIZE = 8
IMG_SIZE = 224

COORD_ACTIONS = {"click", "move"}
SCROLL_TO_IDX = {"up": 0, "down": 1}
SCROLL_NONE = 2


# ------------------------------------------------------------------ discovery ---
def load_demos(data_root, apps):
    """收集 DemoSample v2 文件；返回带 _source / _app / _task 的记录。"""
    files = sorted(data_root.rglob("demo.json")) + sorted(data_root.glob("demo_*.json"))
    rows, skipped = [], []
    for path in files:
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            skipped.append({"file": str(path), "reason": f"json: {exc}"})
            continue
        if not isinstance(items, list):
            items = [items]
        for item in items:
            app = (
                (item.get("meta") or {}).get("app")
                or (item.get("context") or {}).get("app")
                or path.parent.name
            )
            if app not in apps:
                continue
            task = str(item.get("id", "")).split(":")[0] or path.parent.name
            shot = Path(item.get("screenshot", ""))
            if not shot.is_absolute():
                shot = ROOT / shot
            if not item.get("instruction") or item.get("action") not in ACTION_TO_IDX:
                skipped.append({"file": str(path), "id": item.get("id"), "reason": "bad fields"})
                continue
            if not shot.exists():
                skipped.append({"file": str(path), "id": item.get("id"), "reason": "missing screenshot"})
                continue
            row = dict(item, _source=str(path.resolve().relative_to(ROOT)), _app=app, _task=task,
                       _screenshot=str(shot))
            rows.append(row)
    return rows, skipped


def coord_target(item):
    """屏无关坐标；源数据只有 target_rect 时取控件中心。"""
    nc = item.get("norm_coord")
    if nc and len(nc) == 2:
        return float(nc[0]), float(nc[1])
    params = item.get("params") or {}
    screen = item.get("screen") or [2560, 1440]
    if "x" in params and "y" in params and screen[0] and screen[1]:
        return params["x"] / screen[0], params["y"] / screen[1]
    rect = item.get("target_rect")
    if rect and len(rect) == 4 and screen[0] and screen[1]:
        return ((rect[0] + rect[2]) / 2) / screen[0], ((rect[1] + rect[3]) / 2) / screen[1]
    return None


def split_by_task(rows, val_ratio, seed):
    """按任务切分，保证同一屏/同链不出现在 train 与 val 两侧。"""
    rng = np.random.RandomState(seed)
    tasks = sorted({r["_task"] for r in rows})
    rng.shuffle(tasks)
    n_val = max(1, int(round(len(tasks) * val_ratio))) if val_ratio > 0 and len(tasks) > 1 else 0
    val_tasks = set(tasks[:n_val])
    train_rows = [r for r in rows if r["_task"] not in val_tasks]
    val_rows = [r for r in rows if r["_task"] in val_tasks]
    return train_rows, val_rows, sorted(val_tasks)


# -------------------------------------------------------------------- dataset ---
class DemoDataset(Dataset):
    def __init__(self, rows, tokenizer, image_processor, max_len=MAX_LEN):
        self.rows = rows
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        img = Image.open(item["_screenshot"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        pixel = self.image_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        enc = self.tokenizer(
            item["instruction"].strip(),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        raw_action = item["action"]
        action = ACTION_TO_IDX[raw_action]
        xy = coord_target(item)
        has_xy = raw_action in COORD_ACTIONS and xy is not None
        coords = torch.tensor(xy if has_xy else (0.0, 0.0), dtype=torch.float)

        direction = (item.get("params") or {}).get("direction")
        scroll = SCROLL_TO_IDX.get(direction, SCROLL_NONE)

        return {
            "image": pixel,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "action": torch.tensor(action, dtype=torch.long),
            "coords": coords,
            "coord_mask": torch.tensor(has_xy, dtype=torch.bool),
            "scroll": torch.tensor(scroll, dtype=torch.long),
        }


def class_weights(rows):
    counts = np.ones(len(ACTION_MAP), dtype=np.float64)
    for r in rows:
        counts[ACTION_TO_IDX[r["action"]]] += 1
    counts -= 1
    counts = np.maximum(counts, 1)
    return counts, counts.sum() / (len(ACTION_MAP) * counts)


def sampler_weights(rows):
    counts, _ = class_weights(rows)
    return [1.0 / counts[ACTION_TO_IDX[r["action"]]] for r in rows]


# --------------------------------------------------------------------- eval ---
def select_score(metrics):
    """模型选择：动作准确率优先，但同等动作水平下惩罚坐标误差（避免选到坐标头还没学的早期 epoch）。"""
    acc = metrics.get("action_acc", 0.0) or 0.0
    mae = metrics.get("coord_mae_norm")
    return acc - 0.5 * (mae if mae is not None else 1.0)


def evaluate(model, rows, device, tokenizer, image_processor, batch_size=BATCH_SIZE):
    if not rows:
        return {"n": 0}
    ds = DemoDataset(rows, tokenizer, image_processor)
    loader = DataLoader(ds, batch_size=batch_size)
    model.eval()
    total = correct = scroll_total = scroll_ok = coord_n = 0
    coord_err = []
    hit = hit_n = 0
    per_class = {name: {"n": 0, "correct": 0} for name in ACTION_MAP.values()}
    samples = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images, batch["input_ids"].to(device), batch["attention_mask"].to(device))
            pred = outputs["action_logits"].argmax(1).cpu()
            labels = batch["action"]
            total += labels.numel()
            correct += int((pred == labels).sum())
            for lab, pr in zip(labels.tolist(), pred.tolist()):
                per_class[ACTION_MAP[lab]]["n"] += 1
                per_class[ACTION_MAP[lab]]["correct"] += int(lab == pr)

            scroll_pred = outputs["scroll_logits"].argmax(1).cpu().tolist()
            coord_pred = outputs["coord_pred"].cpu().tolist()
            batch_rows = rows[offset:offset + labels.numel()]
            for row, lab, pr, cp, sp in zip(batch_rows, labels.tolist(), pred.tolist(),
                                            coord_pred, scroll_pred):
                screen = row.get("screen") or [2560, 1440]
                entry = {
                    "id": row.get("id"),
                    "instruction": row.get("instruction"),
                    "action_true": ACTION_MAP[lab],
                    "action_pred": ACTION_MAP[pr],
                    "action_ok": lab == pr,
                    "xy_pred_px": [round(cp[0] * screen[0], 1), round(cp[1] * screen[1], 1)],
                    "scroll_pred": sp,
                }
                rect = row.get("target_rect")
                if rect:
                    entry["target_rect"] = rect
                    cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
                    entry["dist_center_px"] = round(
                        ((cp[0] * screen[0] - cx) ** 2 + (cp[1] * screen[1] - cy) ** 2) ** 0.5, 1)
                if len(samples) < 200:
                    samples.append(entry)

            scroll_labels = batch["scroll"]
            has_scroll = scroll_labels != SCROLL_NONE
            if has_scroll.any():
                spred = outputs["scroll_logits"].argmax(1).cpu()
                scroll_total += int(has_scroll.sum())
                scroll_ok += int((spred[has_scroll] == scroll_labels[has_scroll]).sum())

            mask = batch["coord_mask"].bool()
            if mask.any():
                cpred = outputs["coord_pred"].cpu()[mask]
                ctrue = batch["coords"][mask]
                err = (cpred - ctrue).norm(dim=1)
                coord_err.extend(err.tolist())
                coord_n += int(mask.sum())
                for row, keep, p in zip(batch_rows, mask.tolist(), cpred.tolist()):
                    if not keep:
                        continue
                    rect = row.get("target_rect")
                    if not rect:
                        continue
                    screen = row.get("screen") or [2560, 1440]
                    x, y = p[0] * screen[0], p[1] * screen[1]
                    hit_n += 1
                    hit += int(rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3])
            offset += labels.numel()

    out = {
        "n": total,
        "action_acc": correct / max(total, 1),
        "per_class": per_class,
        "coord_n": coord_n,
        "coord_mae_norm": float(np.mean(coord_err)) if coord_err else None,
        "coord_med_norm": float(np.median(coord_err)) if coord_err else None,
        "coord_med_px": float(np.median(coord_err)) * 2560 if coord_err else None,
        "hit_n": hit_n,
        "hit_rate": (hit / hit_n) if hit_n else None,
        "scroll_n": scroll_total,
        "scroll_acc": (scroll_ok / scroll_total) if scroll_total else None,
        "samples": samples,
    }
    out["majority_baseline"] = majority_baseline(rows)
    return out


def majority_baseline(rows):
    if not rows:
        return None
    counts = {}
    for r in rows:
        counts[r["action"]] = counts.get(r["action"], 0) + 1
    best = max(counts.values())
    return {"action": max(counts, key=counts.get), "acc": best / len(rows)}


# --------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", default="notepad", help="逗号分隔：notepad,qgis,probe")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--base-ckpt", default=str(BASE_CKPT))
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0,
                    help="梯度裁剪范数；不裁剪时坐标头会迅速饱和到 (0,1) 并卡死")
    ap.add_argument("--val-ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    apps = [a.strip() for a in args.apps.split(",") if a.strip()]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    clip = float(args.clip)

    rows, skipped = load_demos(Path(args.data_root), apps)
    if not rows:
        print(f"[data] 没有可用示范（apps={apps}）；跳过：{json.dumps(skipped[:5], ensure_ascii=False)}")
        return 1

    train_rows, val_rows, val_tasks = split_by_task(rows, args.val_ratio, args.seed)
    if not train_rows:
        print("[data] train 为空（任务数不足）")
        return 1

    print(f"[env] device={device} torch={torch.__version__}")
    print(f"[data] apps={apps} n={len(rows)} train={len(train_rows)} val={len(val_rows)}")
    print(f"[data] val_tasks={val_tasks}")

    tokenizer = get_tokenizer()
    image_processor = get_image_processor()
    train_ds = DemoDataset(train_rows, tokenizer, image_processor)
    print(f"[data] 动作分布={ {k: v for k, v in sorted(
        {a: sum(1 for r in train_rows if r['action'] == a) for a in ACTION_MAP.values()}.items()
    ) if v} }")

    model = ComputerUseModelV3(num_actions=len(ACTION_MAP), freeze_vit=True, freeze_gpt2=True)
    base = Path(args.base_ckpt)
    if base.exists():
        state = torch.load(base, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[ckpt] 载入 {base.name} missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[ckpt] 基线不存在，随机初始化：{base}")
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] 全部={sum(p.numel() for p in model.parameters()):,} "
          f"可训练={sum(p.numel() for p in trainable):,}")

    counts, weights = class_weights(train_rows)
    action_weights = torch.tensor(weights, dtype=torch.float, device=device)
    optimizer = optim.Adam(trainable, lr=args.lr)
    sampler = WeightedRandomSampler(sampler_weights(train_rows), num_samples=len(train_rows),
                                    replacement=True)

    history = []
    best = {"score": -1e9, "epoch": 0, "state": None}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = total = 0
        for batch in DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler):
            images = batch["image"].to(device)
            outputs = model(images, batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = model.compute_loss(outputs, batch, action_weights=action_weights)

            optimizer.zero_grad()
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, clip)
            optimizer.step()

            total_loss += loss.detach().item() * batch["action"].numel()
            pred = outputs["action_logits"].argmax(1).cpu()
            correct += int((pred == batch["action"]).sum())
            total += batch["action"].numel()

        train_acc = correct / max(total, 1)
        val = evaluate(model, val_rows, device, tokenizer, image_processor) if val_rows else {"n": 0}
        val_acc = val.get("action_acc", 0.0)
        history.append({"epoch": epoch, "loss": total_loss / max(total, 1),
                        "train_acc": train_acc, "val_action_acc": val_acc,
                        "val_coord_mae": val.get("coord_mae_norm"), "val_n": val.get("n", 0)})
        print(f"Epoch {epoch}: loss={history[-1]['loss']:.4f} train_acc={train_acc:.3f} "
              f"val_acc={val_acc:.3f} val_coord={val.get('coord_mae_norm')} (n={val.get('n', 0)})")
        score = select_score(val) if val_rows else -total_loss / max(total, 1)
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "val_action_acc": val_acc,
                    "val_coord_mae": val.get("coord_mae_norm"),
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    state = best["state"] if best["state"] is not None else model.state_dict()
    model.load_state_dict(state)
    out = Path(args.out) if args.out else CKPT_DIR / f"inloop_{'_'.join(apps)}.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)

    meta = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "apps": apps,
        "base_ckpt": str(base),
        "out": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
        "n_total": len(rows), "n_train": len(train_rows), "n_val": len(val_rows),
        "val_tasks": val_tasks,
        "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
        "best_epoch": best["epoch"], "best_val_action_acc": best["val_action_acc"],
        "train_action_counts": {a: sum(1 for r in train_rows if r["action"] == a) for a in ACTION_MAP.values()},
        "val_action_counts": {a: sum(1 for r in val_rows if r["action"] == a) for a in ACTION_MAP.values()},
        "history": history,
        "val_metrics": evaluate(model, val_rows, device, tokenizer, image_processor) if val_rows else {"n": 0},
        "skipped": skipped[:20],
        "skipped_n": len(skipped),
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out}\n[out] {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
