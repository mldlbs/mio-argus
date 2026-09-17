"""
VM 端到端回归测试 - v3 模型
运行：pytest test_vm_e2e.py -v
"""
import json
import pytest
import torch
from pathlib import Path

from mio_argus import (
    ComputerUseModelV3,
    ACTION_MAP,
    ACTION_TO_IDX,
    get_tokenizer,
    get_image_processor,
)
from test_in_vm_v3 import (
    normalize_path,
    preprocess_image,
    encode_instruction,
    parse_gt_coord,
    parse_gt_scroll,
)

# 基线阈值（来自首次跑分）
BASELINE = {
    "action_accuracy": 0.94,
    "coord_mae": 0.09,
    "scroll_accuracy": 0.88,
    "min_per_class_acc": 0.65,  # 每类别最低准确率
}

DATA_JSON = Path("computer_use_data_real_balanced/balanced_training_data.json")
MODEL_PATH = "computer_use_model_v3_real.pth"
MAX_LEN = 64


@pytest.fixture(scope="module")
def model_and_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from test_in_vm_v3 import main as _  # ensure imports work
    from mio_argus import get_tokenizer, get_image_processor
    from mio_argus import ComputerUseModelV3

    tokenizer = get_tokenizer()
    image_processor = get_image_processor()
    model = ComputerUseModelV3(num_actions=6, freeze_vit=True, freeze_gpt2=True).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    with open(DATA_JSON, encoding="utf-8") as f:
        data = json.load(f)

    return model, tokenizer, image_processor, data, device


def run_vm_inference(model, tokenizer, image_processor, data, device):
    """复用 test_in_vm_v3.py 的推理逻辑，返回聚合指标"""
    from test_in_vm_v3 import (
        normalize_path,
        preprocess_image,
        encode_instruction,
        parse_gt_coord,
        parse_gt_scroll,
    )
    from pathlib import Path
    from PIL import Image
    import numpy as np

    total = 0
    action_correct = 0
    coord_mae_sum = 0.0
    scroll_correct = 0
    per_action = {name: {"total": 0, "correct": 0} for name in ACTION_MAP.values()}
    coord_maes = []

    for item in data:
        img_path = normalize_path(item["screenshot"])
        if not Path(img_path).exists():
            continue

        expected_action = item["action"]
        gt_coord = parse_gt_coord(item["params"])
        gt_scroll = parse_gt_scroll(item["params"])

        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess_image(img, image_processor).to(device)
        except Exception:
            continue

        text_enc = encode_instruction(tokenizer, item["instruction"])
        input_ids = text_enc["input_ids"].to(device)
        attention_mask = text_enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(img_tensor, input_ids, attention_mask)

            action_logits = outputs["action_logits"]
            probs = torch.softmax(action_logits, dim=-1)
            pred_id = int(torch.argmax(probs, dim=-1).item())
            pred_action = ACTION_MAP[pred_id]

            pred_coord = outputs["coord_pred"][0].cpu().numpy()
            scroll_logits = outputs["scroll_logits"]
            pred_scroll = int(torch.argmax(scroll_logits, dim=-1).item())

        total += 1
        ok_action = (pred_action == expected_action)
        action_correct += int(ok_action)

        coord_mae = np.mean(np.abs(pred_coord - gt_coord))
        coord_mae_sum += coord_mae
        coord_maes.append(coord_mae)

        ok_scroll = (pred_scroll == gt_scroll)
        scroll_correct += int(ok_scroll)

        # Per-action stats
        stat = per_action[expected_action]
        stat["total"] += 1
        stat["correct"] += int(ok_action)

    # Return aggregated metrics
    return {
        "total": total,
        "action_correct": action_correct,
        "coord_mae_sum": coord_mae_sum,
        "coord_maes": coord_maes,
        "scroll_correct": scroll_correct,
        "per_action": per_action,
    }


def test_vm_e2e_metrics(model_and_data):
    model, tokenizer, image_processor, data, device = model_and_data

    final = run_vm_inference(model, tokenizer, image_processor, data, device)

    assert final is not None, "No valid samples"
    total = final["total"]

    # Overall metrics
    action_acc = final["action_correct"] / total
    coord_mae = final["coord_mae_sum"] / total
    scroll_acc = final["scroll_correct"] / total

    print(f"\nVM E2E: action_acc={action_acc:.4f}, coord_mae={coord_mae:.6f}, scroll_acc={scroll_acc:.4f}")

    # Assertions
    assert action_acc >= BASELINE["action_accuracy"], \
        f"Action accuracy {action_acc:.4f} < {BASELINE['action_accuracy']}"

    assert coord_mae <= BASELINE["coord_mae"], \
        f"Coord MAE {coord_mae:.6f} > {BASELINE['coord_mae']}"

    assert scroll_acc >= BASELINE["scroll_accuracy"], \
        f"Scroll accuracy {scroll_acc:.4f} < {BASELINE['scroll_accuracy']}"

    # Per-class minimum
    for name, stat in final["per_action"].items():
        if stat["total"] > 0:
            acc = stat["correct"] / stat["total"]
            assert acc >= BASELINE["min_per_class_acc"], \
                f"类别 {name} 准确率 {acc:.4f} < {BASELINE['min_per_class_acc']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])