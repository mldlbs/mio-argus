"""
Computer Use Model v3 评测脚本
- 支持加载 split.json 划分的 test 集
- 计算 Action Accuracy, Macro-F1, Coord MAE, Scroll Accuracy
- 输出混淆矩阵与类别级指标
"""
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader

from mio_argus import (
    ComputerUseModelV3,
    ACTION_MAP,
    ACTION_TO_IDX,
    get_image_processor,
    get_tokenizer,
)
from train_computer_use_v3_real import RealDatasetV3


def load_split(split_path):
    with open(split_path, encoding="utf-8") as f:
        return json.load(f)


def load_data(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def evaluate(model, loader, device):
    model.eval()
    act_preds, act_gts = [], []
    coord_preds, coord_gts = [], []
    scroll_preds, scroll_gts = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(images, input_ids, attention_mask)

            # Action
            action_logits = outputs["action_logits"]
            pred_action = action_logits.argmax(-1).cpu().numpy()
            gt_action = batch["action"].cpu().numpy()
            act_preds.extend(pred_action)
            act_gts.extend(gt_action)

            # Coord
            coord_pred = outputs["coord_pred"].cpu().numpy()
            coord_gt = batch["coords"].cpu().numpy()
            coord_preds.append(coord_pred)
            coord_gts.append(coord_gt)

            # Scroll
            scroll_logits = outputs["scroll_logits"]
            pred_scroll = scroll_logits.argmax(-1).cpu().numpy()
            gt_scroll = batch["scroll"].cpu().numpy()
            scroll_preds.extend(pred_scroll)
            scroll_gts.extend(gt_scroll)

    coord_preds = np.vstack(coord_preds)
    coord_gts = np.vstack(coord_gts)

    # Metrics
    action_acc = accuracy_score(act_gts, act_preds)
    action_f1 = f1_score(act_gts, act_preds, average="macro")
    coord_mae = mean_absolute_error(coord_gts, coord_preds)
    scroll_acc = accuracy_score(scroll_gts, scroll_preds)

    # Per-class action metrics
    per_class = {}
    for i, name in ACTION_MAP.items():
        mask = np.array(act_gts) == i
        if mask.sum() > 0:
            per_class[name] = {
                "accuracy": accuracy_score(np.array(act_gts)[mask], np.array(act_preds)[mask]),
                "count": int(mask.sum()),
            }
        else:
            per_class[name] = {"accuracy": 0.0, "count": 0}

    # Confusion matrix
    cm = confusion_matrix(act_gts, act_preds, labels=list(range(6)))

    return {
        "action_accuracy": action_acc,
        "action_macro_f1": action_f1,
        "coord_mae": coord_mae,
        "scroll_accuracy": scroll_acc,
        "per_class_action": per_class,
        "confusion_matrix": cm.tolist(),
        "action_labels": [ACTION_MAP[i] for i in range(6)],
        "n_samples": len(act_gts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="computer_use_model_v3_real.pth", help="模型权重路径"
    )
    parser.add_argument(
        "--data",
        default="computer_use_data_real_balanced/balanced_training_data.json",
        help="原始数据 JSON",
    )
    parser.add_argument(
        "--split",
        default="computer_use_data_real_balanced/split.json",
        help="split.json 路径",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="eval_results.json", help="结果输出文件")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[eval] device={device}")

    # Load split
    split = load_split(args.split)
    test_indices = split["test_indices"]
    print(f"[eval] test samples: {len(test_indices)}")

    # Load data & model
    data = load_data(args.data)
    tokenizer = get_tokenizer()
    image_processor = get_image_processor()

    test_dataset = RealDatasetV3(data, tokenizer, image_processor, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = ComputerUseModelV3(num_actions=6, freeze_vit=True, freeze_gpt2=True).to(device)
    state = torch.load(args.model, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    print(f"[eval] model loaded from {args.model}")

    # Evaluate
    results = evaluate(model, test_loader, device)

    # Print summary
    print("\n=== Evaluation Results ===")
    print(f"Samples:            {results['n_samples']}")
    print(f"Action Accuracy:    {results['action_accuracy']:.4f}")
    print(f"Action Macro-F1:    {results['action_macro_f1']:.4f}")
    print(f"Coord MAE:          {results['coord_mae']:.6f}")
    print(f"Scroll Accuracy:    {results['scroll_accuracy']:.4f}")
    print("\nPer-class Action Accuracy:")
    for name, metrics in results["per_class_action"].items():
        print(f"  {name:8s}: {metrics['accuracy']:.4f} (n={metrics['count']})")
    print("\nConfusion Matrix (rows=gt, cols=pred):")
    print("       " + "  ".join(f"{c:>6s}" for c in results["action_labels"]))
    for i, row in enumerate(results["confusion_matrix"]):
        print(f"{results['action_labels'][i]:6s} " + "  ".join(f"{v:6d}" for v in row))

    # Save JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()