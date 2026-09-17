"""
在隔离环境(虚拟机/容器)中测试 Computer Use Model
- 读取 advanced_training_data.json
- 对每张截图做预测
- 与真实动作对比，统计准确率
"""
import os
import json
import numpy as np
import torch
from PIL import Image
from model_computer_use import SimpleComputerUseModel

DATA_JSON = os.path.join("computer_use_data_advanced", "advanced_training_data.json")
MODEL_PATH = "computer_use_model_advanced.pth"

ACTION_MAP = {0: "click", 1: "type", 2: "scroll", 3: "move", 4: "hotkey", 5: "press"}
ACTION_TO_IDX = {v: k for k, v in ACTION_MAP.items()}


def normalize_path(p: str) -> str:
    """把 Windows 反斜杠路径转换为当前系统分隔符"""
    return p.replace("\\", os.sep).replace("/", os.sep)


def load_image(path: str):
    image = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")
    print(f"[env] torch={torch.__version__}")

    if not os.path.exists(MODEL_PATH):
        print(f"[error] 模型文件不存在: {MODEL_PATH}")
        return
    if not os.path.exists(DATA_JSON):
        print(f"[error] 数据文件不存在: {DATA_JSON}")
        return

    model = SimpleComputerUseModel(num_actions=6).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("[model] 加载完成")

    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    correct = 0
    missing = 0
    per_action = {}

    for item in data:
        img_path = normalize_path(item["screenshot"])
        if not os.path.exists(img_path):
            missing += 1
            continue

        expected = item["action"].split("(")[0]
        image_tensor = load_image(img_path).to(device)
        text_ids = torch.zeros(1, 10, dtype=torch.long).to(device)

        with torch.no_grad():
            outputs = model(image_tensor, text_ids)
            probs = torch.softmax(outputs["action_logits"], dim=-1)
            pred_id = int(torch.argmax(probs, dim=-1).item())
            pred = ACTION_MAP[pred_id]

        total += 1
        ok = (pred == expected)
        correct += int(ok)

        stat = per_action.setdefault(expected, {"total": 0, "correct": 0})
        stat["total"] += 1
        stat["correct"] += int(ok)

    print("\n==================== 测试结果 ====================")
    if total == 0:
        print("没有找到可用截图，无法测试")
        return
    print(f"样本总数: {total} (缺失截图 {missing})")
    print(f"动作准确率: {correct}/{total} = {100.0 * correct / total:.2f}%")
    print("\n分类别准确率:")
    for action, stat in sorted(per_action.items()):
        acc = 100.0 * stat["correct"] / stat["total"]
        print(f"  {action:<8} {stat['correct']}/{stat['total']} = {acc:.1f}%")
    print("=================================================")


if __name__ == "__main__":
    test()
