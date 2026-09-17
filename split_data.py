"""
将 balanced_training_data.json 按 7:2:1 分层划分为 train/val/test，
生成固定的 split.json 供后续训练/评测复用。
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_JSON = Path("computer_use_data_real_balanced/balanced_training_data.json")
OUT_JSON = Path("computer_use_data_real_balanced/split.json")
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1  # 剩余


def stratified_split(data, train_ratio=0.7, val_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    action_to_indices = defaultdict(list)
    for idx, item in enumerate(data):
        action_to_indices[item["action"]].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for action, indices in action_to_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(indices[:n_train].tolist())
        val_idx.extend(indices[n_train:n_train + n_val].tolist())
        test_idx.extend(indices[n_train + n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def main():
    with open(DATA_JSON, encoding="utf-8") as f:
        data = json.load(f)

    train_idx, val_idx, test_idx = stratified_split(
        data, TRAIN_RATIO, VAL_RATIO, SEED
    )

    split = {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "stats": {
            "total": len(data),
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    print(f"Split saved to {OUT_JSON}")
    print(f"  train: {len(train_idx)} ({len(train_idx)/len(data)*100:.1f}%)")
    print(f"  val:   {len(val_idx)} ({len(val_idx)/len(data)*100:.1f}%)")
    print(f"  test:  {len(test_idx)} ({len(test_idx)/len(data)*100:.1f}%)")


if __name__ == "__main__":
    main()