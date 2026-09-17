"""
在真实均衡数据上训练 Computer Use Model v2
- 适配 collect_data_balanced_real.py 的数据格式
- 使用指令条件化 + 类别加权 + 均衡采样
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from model_computer_use import SimpleComputerUseModel

DATA_DIR = "computer_use_data_real_balanced"
DATASET_JSON = os.path.join(DATA_DIR, "balanced_training_data.json")
MODEL_OUT = "computer_use_model_real_v2.pth"
MAX_LEN = 16

ACTION_MAP = {0: "click", 1: "type", 2: "scroll", 3: "move", 4: "hotkey", 5: "press"}
ACTION_TO_IDX = {v: k for k, v in ACTION_MAP.items()}


class RealDataset(Dataset):
    def __init__(self, data, vocab, indices):
        self.data = [data[i] for i in indices]
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def _encode(self, instr):
        ids = [self.vocab.get(t, 1) for t in instr.lower().split()][:16]
        ids += [0] * (16 - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["screenshot"]
        if not os.path.exists(img_path):
            # fallback: create blank image
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        else:
            img = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        image = torch.from_numpy(arr).permute(2, 0, 1)

        action = ACTION_TO_IDX[item["action"]]
        p = item["params"]
        coords = torch.tensor([
            p.get("x", 0) / 1920.0,
            p.get("y", 0) / 1080.0,
        ], dtype=torch.float)
        scroll_up = 1 if p.get("direction") == "up" else 0

        return {
            "image": image,
            "text_ids": self._encode(item["instruction"]),
            "action": torch.tensor(ACTION_TO_IDX[item["action"]], dtype=torch.long),
            "coords": coords,
            "scroll_up": torch.tensor(scroll_up, dtype=torch.long),
        }


def build_vocab(data):
    vocab = {"<pad>": 0, "<?>": 1}
    for item in data:
        for tok in item["instruction"].lower().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def compute_weights(dataset):
    counts = np.zeros(6)
    for item in dataset.data:
        counts[ACTION_TO_IDX[item["action"]]] += 1
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (6.0 * counts)
    return counts, weights


def evaluate(model, loader, device):
    model.eval()
    per_class = {a: {"total": 0, "correct": 0} for a in ACTION_MAP.values()}
    total, correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            text_ids = batch["text_ids"].to(device)
            labels = batch["action"].to(device)
            out = model(images, text_ids)
            pred = out["action_logits"].argmax(dim=-1)
            for t, p in zip(labels.tolist(), pred.tolist()):
                name = ACTION_MAP[t]
                per_class[name]["total"] += 1
                per_class[name]["correct"] += int(t == p)
                total += 1
                correct += int(t == p)
    return correct / max(total, 1), per_class


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}, torch={torch.__version__}")

    with open("computer_use_data_real_balanced/balanced_training_data.json", encoding="utf-8") as f:
        data = json.load(f)

    vocab = build_vocab(data)
    print(f"[vocab] size={len(vocab)}")

    # 固定划分 train/val (每类分层)
    rng = np.random.RandomState(42)
    class_indices = {a: [] for a in ACTION_TO_IDX.values()}
    for i, item in enumerate(data):
        class_indices[ACTION_TO_IDX[item["action"]]].append(i)
    train_idx, val_idx = [], []
    for a, idxs in class_indices.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        split = int(len(idxs) * 0.8)
        train_idx += idxs[:split].tolist()
        val_idx += idxs[split:].tolist()

    train_ds = RealDataset(data, vocab, train_idx)
    val_ds = RealDataset(data, vocab, val_idx)

    counts, class_w = compute_weights(train_ds)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")
    print(f"[data] 类别分布={counts.astype(int).tolist()}")
    print(f"[data] 类别权重={np.round(class_w, 3).tolist()}")

    sample_weights = [1.0 / counts[ACTION_TO_IDX[it["action"]]] for it in train_ds.data]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = SimpleComputerUseModel(num_actions=6, vocab_size=len(vocab)).to(device)
    print(f"[model] 参数量={sum(p.numel() for p in model.parameters()):,}")

    ce_weighted = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float, device=device))
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    for ep in range(1, 21):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            text_ids = batch["text_ids"].to(device)
            labels = batch["action"].to(device)
            coords = batch["coords"].to(device)
            scroll = batch["scroll_up"].to(device)

            out = model(images, text_ids)
            loss = ce_weighted(out["action_logits"], labels)
            loss = loss + 0.3 * nn.MSELoss()(out["coord_logits"], coords)
            loss = loss + 0.3 * nn.CrossEntropyLoss()(out["scroll_logits"], scroll)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            n += len(labels)

        acc, per_class = evaluate(model, val_loader, device)
        print(f"[epoch {ep:02d}] loss={total_loss / n:.4f} val_acc={acc * 100:.2f}%")

    print("\n==================== 分类别准确率 ====================")
    acc, per_class = evaluate(model, val_loader, device)
    for name, st in sorted(per_class.items()):
        a = 100.0 * st["correct"] / max(st["total"], 1)
        print(f"  {name:<8} {st['correct']}/{st['total']} = {a:.1f}%")
    print(f"  总准确率: {acc * 100:.2f}%")
    print("=====================================================")

    torch.save({"model": model.state_dict(), "vocab_size": len(vocab)}, "computer_use_model_real_v2.pth")
    print(f"[save] computer_use_model_real_v2.pth")


if __name__ == "__main__":
    train()