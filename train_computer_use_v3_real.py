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
from model_computer_use_v3 import (
    ComputerUseModelV3,
    ComputerUseModelV3Small,
    ACTION_MAP,
    ACTION_TO_IDX,
    get_image_processor,
    get_tokenizer,
)

# 配置
DATASET_JSON = os.path.join("computer_use_data_real_balanced", "balanced_training_data.json")
MODEL_OUT = "computer_use_model_v3_real.pth"
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
        coords = torch.tensor([
            params.get("x", 0) / 1920.0,
            params.get("y", 0) / 1080.0,
        ], dtype=torch.float)
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
            "coord_mask": torch.tensor(True, dtype=torch.bool),
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


def evaluate(model, loader, device):
    model.eval()
    per_class = {action: {"total": 0, "correct": 0} for action in ACTION_MAP.values()}
    total, correct = 0, 0
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
    return correct / max(total, 1), per_class


def train():
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
    print(f"[data] train={len(dataset)} val={len(val_dataset)}")
    print(f"[data] 类别分布={counts.astype(int).tolist()}")
    print(f"[data] 类别权重={np.round(class_weights, 3).tolist()}")

    model = ComputerUseModelV3(num_actions=6, freeze_vit=True, freeze_gpt2=True).to(device)
    print(f"[model] 参数量={sum(p.numel() for p in model.parameters()):,}")
    print(f"[model] 可训练参数={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    action_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
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

        acc = 100.0 * correct / max(total, 1)
        val_acc, val_per_class = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}: loss={total_loss / len(dataset):.4f}, "
            f"acc={acc:.2f}%, val_acc={val_acc * 100:.2f}%"
        )
        print(f"[val] 类别准确率={val_per_class}")

    torch.save(model.state_dict(), MODEL_OUT)
    print("模型已保存")


if __name__ == "__main__":
    train()
