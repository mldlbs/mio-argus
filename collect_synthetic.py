"""
合成数据生成器 - 用 PIL 生成逼真的 UI 截图 + 随机动作标注
无需 Xvfb/pyautogui，纯 CPU 运算
"""
import json
import random
import os
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

DATA_DIR = Path("computer_use_data_real_balanced")
DATA_DIR.mkdir(exist_ok=True)

ACTIONS = ["click", "type", "scroll", "move", "hotkey", "press"]
W, H = 1920, 1080


def _random_color(min_val=50, max_val=255):
    return tuple(random.randint(min_val, max_val) for _ in range(3))


def _draw_button(draw, x, y, w, h, text="", active=False):
    color = _random_color(100, 200) if active else _random_color(180, 240)
    draw.rectangle([x, y, x + w, y + h], fill=color, outline="black", width=2)
    if text:
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x + (w - tw) // 2, y + (h - th) // 2), text, fill="black", font=font)


def _draw_text_input(draw, x, y, w, h, text=""):
    draw.rectangle([x, y, x + w, y + h], fill="white", outline="gray", width=2)
    if text:
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        draw.text((x + 5, y + 5), text, fill="black", font=font)


def _draw_panel(draw, x, y, w, h, title=""):
    draw.rectangle([x, y, x + w, y + h], fill=_random_color(200, 240), outline="black", width=2)
    if title:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        draw.text((x + 10, y + 5), title, fill="black", font=font)


def _draw_scrollbar(draw, x, y, h, position=0.5):
    draw.rectangle([x, y, x + 15, y + h], fill="lightgray", outline="gray")
    handle_h = max(30, h // 5)
    handle_y = y + int(position * (h - handle_h))
    draw.rectangle([x, handle_y, x + 15, handle_y + handle_h], fill="gray")


def generate_random_ui():
    """生成随机 UI 截图"""
    img = Image.new("RGB", (W, H), _random_color(220, 250))
    draw = ImageDraw.Draw(img)

    # 背景 - 模拟桌面
    for _ in range(random.randint(2, 5)):
        x = random.randint(0, W - 400)
        y = random.randint(0, H - 300)
        w = random.randint(200, 600)
        h = random.randint(150, 400)
        _draw_panel(draw, x, y, w, h, title=f"Window {_ + 1}")

    # 按钮
    for _ in range(random.randint(3, 8)):
        x = random.randint(50, W - 200)
        y = random.randint(50, H - 100)
        texts = ["OK", "Cancel", "Submit", "Click", "Next", "Back", "Save", "Open"]
        _draw_button(draw, x, y, random.randint(80, 150), random.randint(30, 50),
                     text=random.choice(texts))

    # 文本输入框
    for _ in range(random.randint(1, 3)):
        x = random.randint(100, W - 300)
        y = random.randint(100, H - 100)
        _draw_text_input(draw, x, y, random.randint(200, 400), random.randint(25, 40),
                         text=random.choice(["", "hello@world.com", "search...", "password"]))

    # 滚动条
    _draw_scrollbar(draw, W - 30, 100, H - 200, position=random.random())

    # 菜单栏
    for i, label in enumerate(["File", "Edit", "View", "Help"]):
        x = i * 80 + 10
        draw.rectangle([x, 0, x + 75, 25], fill=_random_color(200, 230), outline="gray")
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        draw.text((x + 10, 5), label, fill="black", font=font)

    return img


def _collect_click():
    x = random.randint(50, W - 50)
    y = random.randint(50, H - 50)
    return {"x": x, "y": y}


def _collect_type():
    texts = [
        "hello", "test", "computer use", "model training",
        "data collection", "click here", "scroll down",
        "press enter", "select all", "copy text",
        "open file", "save document", "close window",
        "search query", "next page", "email@test.com",
        "password123", "https://example.com", "user_name"
    ]
    return {"text": random.choice(texts)}


def _collect_scroll():
    return {"direction": random.choice(["up", "down"]), "amount": random.randint(1, 5)}


def _collect_move():
    return {"x": random.randint(50, W - 50), "y": random.randint(50, H - 50)}


def _collect_hotkey():
    keys_list = [
        ["ctrl", "c"], ["ctrl", "v"], ["ctrl", "x"],
        ["ctrl", "a"], ["ctrl", "s"], ["ctrl", "z"],
        ["alt", "tab"], ["alt", "f4"], ["ctrl", "shift", "esc"]
    ]
    return {"keys": random.choice(keys_list)}


def _collect_press():
    return {"key": random.choice(["enter", "tab", "escape", "space", "backspace", "delete"])}


ACTION_METHODS = {
    "click": _collect_click,
    "type": _collect_type,
    "scroll": _collect_scroll,
    "move": _collect_move,
    "hotkey": _collect_hotkey,
    "press": _collect_press,
}


def _make_instruction(action, params):
    if action == "click":
        return f"click at {params['x']} {params['y']}"
    if action == "type":
        return f"type {params['text']} in the input"
    if action == "scroll":
        return f"scroll {params['direction']} the page"
    if action == "move":
        return f"move mouse to {params['x']} {params['y']}"
    if action == "hotkey":
        return "press " + " ".join(params["keys"])
    if action == "press":
        return f"press {params['key']}"
    return "unknown"


def collect(target_per_action=84):
    json_path = DATA_DIR / "balanced_training_data.json"
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"[load] 已有 {len(data)} 条样本")
    else:
        data = []

    counts = Counter(item["action"] for item in data)
    print(f"[info] 当前分布: {dict(counts)}")
    print(f"[info] 目标: 每类 {target_per_action} 个")

    need = {a: max(0, target_per_action - counts.get(a, 0)) for a in ACTIONS}
    total_needed = sum(need.values())
    if total_needed == 0:
        print("[info] 已达标，无需采集")
        return

    print(f"[info] 需新增 {total_needed} 条样本")

    actions_to_collect = []
    for a, n in need.items():
        actions_to_collect.extend([a] * n)
    random.shuffle(actions_to_collect)

    try:
        for i, action in enumerate(actions_to_collect, 1):
            # 生成随机 UI 截图
            img = generate_random_ui()
            fname = f"balanced_{action}_{len(data):04d}.png"
            filepath = DATA_DIR / fname
            img.save(str(filepath))

            # 执行动作
            method = ACTION_METHODS[action]
            params = method()
            instruction = _make_instruction(action, params)

            record = {
                "timestamp": datetime.now().isoformat(),
                "screenshot": str(filepath),
                "instruction": instruction,
                "action": action,
                "params": params
            }
            data.append(record)

            if i % 50 == 0:
                print(f"[{i}/{total_needed}] 已采集 {len(data)} 条")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

    except KeyboardInterrupt:
        print("\n[interrupt] 用户中断")

    finally:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[save] 已保存 {len(data)} 条到 {json_path}")

        cnt = Counter(item["action"] for item in data)
        print("\n最终分布:")
        for a in ACTIONS:
            print(f"  {a}: {cnt.get(a, 0)}")


if __name__ == "__main__":
    target = int(os.environ.get("TARGET_PER_ACTION", "84"))
    print("=== 合成数据生成 ===")
    print(f"目标: 每类 {target} 条")
    print("使用 PIL 生成 UI 截图，无需 pyautogui/Xvfb")
    collect(target_per_action=target)
    print("\n生成完成!")