"""
Computer Use Model v3 训练脚本
- ViT + GPT-2 完整架构
- Huber Loss 坐标回归
- Cross-Attention 融合
- 支持合成/真实数据
"""
import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
_sys.path[:0] = [p for p in (str(_ROOT / "mio"), str(_ROOT)) if p not in _sys.path]

from mio_argus import (
    ComputerUseModelV3,
    ComputerUseModelV3Small,
    ACTION_MAP,
    ACTION_TO_IDX,
    get_image_processor,
    get_tokenizer,
)

# 配置
DATASET_JSON = os.path.join("artifacts/computer_use_data_real_balanced", "balanced_training_data.json")
MODEL_OUT = "artifacts/checkpoints/computer_use_model_v3_real.pth"
MAX_LEN = 64
BATCH_SIZE = 8
EPOCHS = 30
LR = 2e-4


class RealDatasetV3(Dataset):
    def __init__(self, data, tokenizer, image_processor, indices, max_len=64):
        self.data = [data[i] for i in indices]
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _encode(self, instr):
        encoding = self.tokenizer(
            instr,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["screenshot"]
        if not os.path.exists(img_path):
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        else:
            img = Image.open(img_path).convert("RGB").resize((224, 224))

        img_inputs = self.image_processor(images=img, return_tensors="pt")
        image = img_inputs["pixel_values"].squeeze(0)

        action = ACTION_TO_IDX[item["action"]]
        params = item["params"]
        # 只有 click/move 带坐标。其余动作（type/scroll/hotkey/press）没有 x/y，
        # 若照旧写成 (0,0) 且 coord_mask 恒为 True，坐标头会被 336/504 个零目标
        # 训练成"输出条件中位数"（见 results/paper/COORD_HEAD_FINDINGS.md）。
        # 与 train_inloop.py 一致：无坐标的样本 coord_mask=False，不参与 coord loss。
        has_xy = item["action"] in ("click", "move") and "x" in params and "y" in params
        coords = torch.tensor([
            params.get("x", 0) / 1920.0,
            params.get("y", 0) / 1080.0,
        ] if has_xy else (0.0, 0.0), dtype=torch.float)
        scroll = 2
        if "direction" in params:
            if params["direction"] == "up":
                scroll = 0
            elif params["direction"] == "down":
                scroll = 1

        text_enc = self._encode(item["instruction"])
        return {
            "image": img_inputs["pixel_values"].squeeze(0),
            "input_ids": text_enc["input_ids"],
            "attention_mask": text_enc["attention_mask"],
            "action": torch.tensor(ACTION_TO_IDX[item["action"]], dtype=torch.long),
            "coords": coords,
            "coord_mask": torch.tensor(has_xy, dtype=torch.bool),
            "scroll": torch.tensor(scroll, dtype=torch.long),
        }


def build_vocab(data):
    vocab = {"<pad>": 0, "<?>": 1}
    for item in data:
        for token in item["instruction"].lower().split():
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def split_data_stratified(data, train_ratio=0.8, seed=42):
    rng = np.random.RandomState(seed)
    class_indices = {action: [] for action in ACTION_TO_IDX}
    for idx, item in enumerate(data):
        class_indices[item["action"]].append(idx)
    train_idx, val_idx = [], []
    for action, indices in class_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        split = int(len(indices) * train_ratio)
        train_idx.extend(indices[:split].tolist())
        val_idx.extend(indices[split:].tolist())
    return train_idx, val_idx


def compute_class_weights(data):
    counts = np.zeros(6)
    for item in data:
        counts[ACTION_TO_IDX[item["action"]]] += 1
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (6.0 * counts)
    return counts, weights


def coord_metrics(pred, gt, mask, hit_tol=0.02):
    """坐标指标：MAE、±tol 命中率、预测/真值标准差比。

    pred/gt: [N,2]（归一坐标）；mask: [N] bool（只有 click/move 为 True）。
    hit_tol=0.02 是"两个轴都在 2% 归一幅度内"，x 方向≈38px@1920。
    std 比 <1 表示输出被收缩（向均值塌），这是 0/84 的直接特征。
    """
    if int(mask.sum()) == 0:
        return None
    p, g = pred[mask], gt[mask]
    err = (p - g).abs()
    return {
        "n": int(mask.sum()),
        "mae_x": err[:, 0].mean().item(),
        "mae_y": err[:, 1].mean().item(),
        "hit_2pct": (err < hit_tol).all(dim=1).float().mean().item(),
        "std_ratio_x": (p[:, 0].std(unbiased=False) /
                        g[:, 0].std(unbiased=False).clamp_min(1e-6)).item(),
        "std_ratio_y": (p[:, 1].std(unbiased=False) /
                        g[:, 1].std(unbiased=False).clamp_min(1e-6)).item(),
    }


def evaluate(model, loader, device):
    model.eval()
    per_class = {action: {"total": 0, "correct": 0} for action in ACTION_MAP.values()}
    total, correct = 0, 0
    preds, gts, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            action_labels = batch["action"].to(device)

            outputs = model(images, input_ids, attention_mask)
            action_logits = outputs["action_logits"]
            _, predicted = action_logits.max(1)
            total += action_labels.size(0)
            correct += predicted.eq(action_labels).sum().item()
            for label, pred in zip(action_labels.tolist(), predicted.tolist()):
                name = ACTION_MAP[label]
                per_class[name]["total"] += 1
                per_class[name]["correct"] += int(label == pred)

            preds.append(outputs["coord_pred"].detach().cpu())
            gts.append(batch["coords"].cpu())
            masks.append(batch["coord_mask"].bool().cpu())
    coord = coord_metrics(torch.cat(preds), torch.cat(gts), torch.cat(masks))
    return correct / max(total, 1), per_class, coord



def fmt_coord(c):
    if not c:
        return "n/a"
    return (f"n={c['n']} MAE x={c['mae_x']:.4f} y={c['mae_y']:.4f} "
            f"hit2%={c['hit_2pct']:.3f} "
            f"std_ratio x={c['std_ratio_x']:.2f} y={c['std_ratio_y']:.2f}")


def train(out_path=MODEL_OUT, epochs=EPOCHS, metrics_out="", note=""):
    import time as _time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}, torch={torch.__version__}")

    with open(DATASET_JSON, encoding="utf-8") as file:
        data = json.load(file)

    tokenizer = get_tokenizer()
    image_processor = get_image_processor()
    rng = np.random.RandomState(42)
    class_indices = {action: [] for action in ACTION_TO_IDX}
    for idx, item in enumerate(data):
        class_indices[item["action"]].append(idx)
    train_idx, val_idx = [], []
    for action, indices in class_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        split = int(len(indices) * 0.8)
        train_idx.extend(indices[:split].tolist())
        val_idx.extend(indices[split:].tolist())

    dataset = RealDatasetV3(data, tokenizer, image_processor, train_idx)
    val_dataset = RealDatasetV3(data, tokenizer, image_processor, val_idx)
    counts, class_weights = compute_class_weights(data)
    n_coord = sum(1 for i in train_idx if data[i]["action"] in ("click", "move"))
    print(f"[data] train={len(dataset)} val={len(val_dataset)}")
    print(f"[data] 类别分布={counts.astype(int).tolist()}")
    print(f"[data] 类别权重={np.round(class_weights, 3).tolist()}")
    print(f"[data] 参与 coord loss 的 train 样本={n_coord}/{len(dataset)}")

    model = ComputerUseModelV3(num_actions=6, freeze_vit=True, freeze_gpt2=True).to(device)
    print(f"[model] 参数量={sum(p.numel() for p in model.parameters()):,}")
    print(f"[model] 可训练参数={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    action_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    history = []
    t0 = _time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        tp, tg, tm = [], [], []
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE, sampler=WeightedRandomSampler(
            [1.0 / counts[ACTION_TO_IDX[data[idx]["action"]]] for idx in train_idx],
            num_samples=len(train_idx),
            replacement=True,
        )):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            action_labels = batch["action"].to(device)

            outputs = model(images, input_ids, attention_mask)
            loss = model.compute_loss(outputs, batch, action_weights=action_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
            _, pred = outputs["action_logits"].max(1)
            total += action_labels.size(0)
            correct += pred.eq(action_labels).sum().item()

            tp.append(outputs["coord_pred"].detach().cpu())
            tg.append(batch["coords"].cpu())
            tm.append(batch["coord_mask"].bool().cpu())

        acc = 100.0 * correct / max(total, 1)
        train_coord = coord_metrics(torch.cat(tp), torch.cat(tg), torch.cat(tm))
        val_acc, val_per_class, val_coord = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}: loss={total_loss / max(len(dataset), 1):.4f}, "
            f"acc={acc:.2f}%, val_acc={val_acc * 100:.2f}%"
        )
        print(f"[train] coord {fmt_coord(train_coord)}")
        print(f"[val] coord {fmt_coord(val_coord)}   ({_time.time() - t0:.0f}s)")
        history.append({"epoch": epoch, "loss": total_loss / max(len(dataset), 1),
                        "acc": acc, "val_acc": val_acc,
                        "train_coord": train_coord, "val_coord": val_coord})

    _Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"[save] {out_path}")

    # 全部 168 条坐标样本 —— 与 COORD_HEAD_FINDINGS.md 的探针同一批，可直接比较
    coord_idx = [i for i in range(len(data)) if data[i]["action"] in ("click", "move")]
    coord_loader = DataLoader(RealDatasetV3(data, tokenizer, image_processor, coord_idx),
                              batch_size=BATCH_SIZE)
    _, _, full_coord = evaluate(model, coord_loader, device)
    print(f"[all168] coord {fmt_coord(full_coord)}")

    if metrics_out:
        payload = {
            "out": out_path,
            "note": note,
            "epochs": epochs,
            "train_size": len(dataset),
            "val_size": len(val_dataset),
            "n_coord_train": n_coord,
            "final_coord_all168": full_coord,
            "history": history,
            "elapsed_sec": round(_time.time() - t0, 1),
            "coord_mask_fix": True,
        }
        p = _Path(metrics_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[metrics] {p}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=MODEL_OUT,
                    help="checkpoint 路径（默认会覆盖正式 checkpoint，重训请显式指定新路径）")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--metrics-out", default="")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    train(out_path=args.out, epochs=args.epochs, metrics_out=args.metrics_out, note=args.note)
