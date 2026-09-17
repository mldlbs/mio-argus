"""
虚拟机内数据采集 - 在 Xvfb 虚拟屏幕上执行 pyautogui 动作并截图
不触碰宿主机的鼠标/键盘
"""
import os
import json
import time
import random
from datetime import datetime
from pathlib import Path
from collections import Counter

import pyautogui
from PIL import Image, ImageDraw, ImageFont

# 关闭 pyautogui 安全限制（虚拟环境，无需担心）
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.3

DATA_DIR = Path("/app/data")
DATA_DIR.mkdir(exist_ok=True)

ACTIONS = ["click", "type", "scroll", "move", "hotkey", "press"]
ACTION_MAP = {
    0: "click", 1: "type", 2: "scroll",
    3: "move", 4: "hotkey", 5: "press"
}


def _make_background():
    """生成随机背景截图（模拟真实屏幕内容）"""
    img = Image.new("RGB", (1920, 1080), (
        random.randint(20, 240),
        random.randint(20, 240),
        random.randint(20, 240)
    ))
    draw = ImageDraw.Draw(img)
    # 画一些随机矩形模拟窗口/按钮
    for _ in range(random.randint(3, 10)):
        x1 = random.randint(0, 1600)
        y1 = random.randint(0, 800)
        x2 = x1 + random.randint(50, 300)
        y2 = y1 + random.randint(30, 200)
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        draw.rectangle([x1, y1, x2, y2], fill=color, outline="black")
    # 画一些随机线条
    for _ in range(random.randint(5, 15)):
        x1 = random.randint(0, 1920)
        y1 = random.randint(0, 1080)
        x2 = random.randint(0, 1920)
        y2 = random.randint(0, 1080)
        draw.line([x1, y1, x2, y2], fill="black", width=1)
    return img


def _capture_virtual_screen():
    """截取虚拟屏幕（Xvfb）"""
    try:
        screenshot = pyautogui.screenshot()
        return screenshot
    except Exception:
        return _make_background()


def _collect_click():
    x = random.randint(100, 1800)
    y = random.randint(100, 980)
    pyautogui.click(x, y)
    return {"x": x, "y": y}


def _collect_type():
    texts = [
        "hello", "test", "computer use", "model training",
        "data collection", "click here", "scroll down",
        "press enter", "select all", "copy text",
        "open file", "save document", "close window",
        "search query", "next page"
    ]
    text = random.choice(texts)
    pyautogui.typewrite(text, interval=0.01)
    return {"text": text}


def _collect_scroll():
    direction = random.choice(["up", "down"])
    amount = random.randint(1, 5)
    pyautogui.scroll(amount if direction == "up" else -amount)
    return {"direction": direction, "amount": amount}


def _collect_move():
    x = random.randint(100, 1800)
    y = random.randint(100, 980)
    pyautogui.moveTo(x, y)
    return {"x": x, "y": y}


def _collect_hotkey():
    keys_list = [
        ["ctrl", "c"], ["ctrl", "v"], ["ctrl", "x"],
        ["ctrl", "a"], ["ctrl", "s"], ["ctrl", "z"],
        ["alt", "tab"], ["alt", "f4"], ["ctrl", "shift", "esc"],
        ["ctrl", "alt", "delete"]
    ]
    keys = random.choice(keys_list)
    pyautogui.hotkey(*keys)
    return {"keys": keys}


def _collect_press():
    key = random.choice(["enter", "tab", "escape", "space", "backspace", "delete"])
    pyautogui.press(key)
    return {"key": key}


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
            # 每次采集前生成新的随机背景
            bg = _make_background()
            bg.save(str(DATA_DIR / "_temp_bg.png"))
            # 用 pyautogui 截取虚拟屏幕
            screenshot = _capture_virtual_screen()

            fname = f"balanced_{action}_{len(data):04d}.png"
            filepath = DATA_DIR / fname
            screenshot.save(str(filepath))

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

            if i % 10 == 0:
                print(f"[{i}/{total_needed}] 已采集 {len(data)} 条")
                # 中间保存
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            time.sleep(0.3)

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
    print("=== 虚拟机内数据采集 ===")
    print(f"目标: 每类 {target} 条")
    print("在 Xvfb 虚拟屏幕上执行，不影响宿主机")
    time.sleep(2)
    collect(target_per_action=target)
    print("\n采集完成!")