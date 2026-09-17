"""
VM 端到端测试 - v3 模型版本
- 使用 MioArgus (ComputerUseModelV3)
- 读取 balanced_training_data.json
- 对每张截图做预测，与真实动作/坐标/滚动对比
- 输出动作准确率、坐标 MAE、滚动准确率
"""
import os
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from mio_argus import (
    ComputerUseModelV3,
    ACTION_MAP,
    ACTION_TO_IDX,
    get_image_processor,
    get_tokenizer,
)

DATA_JSON = Path("computer_use_data_real_balanced/balanced_training_data.json")
MODEL_PATH = "computer_use_model_v3_real.pth"
MAX_LEN = 64


def normalize_path(p: str) -> str:
    return p.replace("\\", os.sep).replace("/", os.sep)


def preprocess_image(img: Image.Image, image_processor) -> torch.Tensor:
    """与训练一致的预处理：使用 get_image_processor() 归一化"""
    img_inputs = image_processor(images=img, return_tensors="pt")
    return img_inputs["pixel_values"]


def encode_instruction(tokenizer, instruction: str) -> dict:
    encoding = tokenizer(
        instruction,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
    }


def parse_gt_coord(params: dict) -> np.ndarray:
    """归一化到 [0,1]，与训练一致"""
    x = params.get("x", 0) / 1920.0
    y = params.get("y", 0) / 1080.0
    return np.array([x, y], dtype=np.float32)


def parse_gt_scroll(params: dict) -> int:
    if "direction" in params:
        return 0 if params["direction"] == "up" else 1
    return 2  # none


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}, torch={torch.__version__}")

    if not Path(MODEL_PATH).exists():
        print(f"[error] 模型不存在: {MODEL_PATH}")
        return
    if not DATA_JSON.exists():
        print(f"[error] 数据不存在: {DATA_JSON}")
        return

    # Load model
    tokenizer = get_tokenizer()
    image_processor = get_image_processor()

    model = ComputerUseModelV3(num_actions=6, freeze_vit=True, freeze_gpt2=True).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"[model] v3 loaded from {MODEL_PATH}")

    # Load data
    with open(DATA_JSON, encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    missing = 0
    action_correct = 0
    coord_mae_sum = 0.0
    scroll_correct = 0

    per_action = {name: {"total": 0, "correct": 0} for name in ACTION_MAP.values()}
    per_action_coord = {name: [] for name in ACTION_MAP.values()}

    for item in data:
        img_path = normalize_path(item["screenshot"])
        if not Path(img_path).exists():
            missing += 1
            continue

        expected_action = item["action"]
        gt_coord = parse_gt_coord(item["params"])
        gt_scroll = parse_gt_scroll(item["params"])

        # Load & preprocess image
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess_image(img, image_processor).to(device)
        except Exception as e:
            print(f"[warn] 图片加载失败 {img_path}: {e}")
            missing += 1
            continue

        # Encode instruction
        text_enc = encode_instruction(tokenizer, item["instruction"])
        input_ids = text_enc["input_ids"].to(device)
        attention_mask = text_enc["attention_mask"].to(device)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor, input_ids, attention_mask)

            # Action
            action_logits = outputs["action_logits"]
            probs = torch.softmax(action_logits, dim=-1)
            pred_id = int(torch.argmax(probs, dim=-1).item())
            pred_action = ACTION_MAP[pred_id]

            # Coord
            pred_coord = outputs["coord_pred"][0].cpu().numpy()

            # Scroll
            scroll_logits = outputs["scroll_logits"]
            pred_scroll = int(torch.argmax(scroll_logits, dim=-1).item())

        # Metrics
        total += 1
        ok_action = (pred_action == expected_action)
        action_correct += int(ok_action)

        coord_mae = np.mean(np.abs(pred_coord - gt_coord))
        coord_mae_sum += coord_mae
        per_action_coord[expected_action].append(coord_mae)

        ok_scroll = (pred_scroll == gt_scroll)
        scroll_correct += int(ok_scroll)

        # Per-action stats
        stat = per_action[expected_action]
        stat["total"] += 1
        stat["correct"] += int(ok_action)

    # Summary
    print("\n==================== VM 端到端测试 (v3) ====================")
    if total == 0:
        print("无可用样本")
        return

    print(f"样本总数: {total} (缺失/失败 {missing})")
    print(f"动作准确率: {action_correct}/{total} = {100.0 * action_correct / total:.2f}%")
    print(f"坐标 MAE:   {coord_mae_sum / total:.6f}")
    print(f"滚动准确率: {scroll_correct}/{total} = {100.0 * scroll_correct / total:.2f}%")

    print("\n分动作准确率:")
    for action, stat in sorted(per_action.items()):
        if stat["total"] > 0:
            acc = 100.0 * stat["correct"] / stat["total"]
            mae = np.mean(per_action_coord[action]) if per_action_coord[action] else 0.0
            print(f"  {action:<8} {stat['correct']}/{stat['total']} = {acc:.1f}%  |  坐标 MAE: {mae:.6f}")

    print("=============================================================")


if __name__ == "__main__":
    main()