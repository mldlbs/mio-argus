"""
增量采集：在现有 balanced_training_data.json 基础上继续采集，
直到每类达到 target_per_action（默认 84，共 504 样本）。
"""
import pyautogui
import json
import time
import os
import random
from datetime import datetime
from collections import Counter
from pathlib import Path

# --- 动作执行函数 ---
def _collect_click():
    x = random.randint(200, 1700)
    y = random.randint(200, 900)
    pyautogui.click(x, y)
    return {"x": x, "y": y}

def _collect_type():
    text = random.choice(["hello", "test", "computer use", "model training", "data collection"])
    pyautogui.typewrite(text, interval=0.01)
    return {"text": text}

def _collect_scroll():
    direction = random.choice(["up", "down"])
    amount = random.randint(1, 3)
    pyautogui.scroll(amount if direction == "up" else -amount)
    return {"direction": direction, "amount": 1}

def _collect_move():
    x = random.randint(200, 1700)
    y = random.randint(200, 900)
    pyautogui.moveTo(x, y)
    return {"x": x, "y": y}

def _collect_hotkey():
    keys = random.choice([["ctrl", "c"], ["ctrl", "v"], ["ctrl", "x"], ["alt", "tab"], ["ctrl", "s"]])
    pyautogui.hotkey(*keys)
    return {"keys": keys}

def _collect_press():
    key = random.choice(["enter", "tab", "escape", "space"])
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

ACTIONS = list(ACTION_METHODS.keys())

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


class BalancedDataCollector:
    def __init__(self, save_dir="computer_use_data_real_balanced"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.data = []
        self.action_count = 0

    def load_existing(self, json_path):
        if Path(json_path).exists():
            with open(json_path, encoding="utf-8") as f:
                self.data = json.load(f)
            print(f"[load] 已有 {len(self.data)} 条样本")
        else:
            print("[load] 无现有数据，从头开始")

    def current_counts(self):
        return Counter(item["action"] for item in self.data)

    def take_screenshot(self, filename):
        screenshot = pyautogui.screenshot()
        filepath = self.save_dir / filename
        screenshot.save(filepath)
        return str(filepath)

    def record(self, action, params, screenshot_path):
        instruction = _make_instruction(action, params)
        record = {
            "timestamp": datetime.now().isoformat(),
            "screenshot": screenshot_path,
            "instruction": instruction,
            "action": action,
            "params": params
        }
        self.data.append(record)
        self.action_count += 1
        return record

    def collect_to_target(self, target_per_action=84):
        counts = self.current_counts()
        print(f"[info] 当前分布: {dict(counts)}")
        print(f"[info] 目标: 每类 {target_per_action} 个")

        # 计算每类还需采集数量
        need = {a: max(0, target_per_action - counts.get(a, 0)) for a in ACTIONS}
        total_needed = sum(need.values())
        if total_needed == 0:
            print("[info] 已达标，无需采集")
            return

        print(f"[info] 需新增 {total_needed} 条样本")
        print("按 Ctrl+C 可随时中断，进度会自动保存")

        # 构建待采集动作列表并打乱
        actions_to_collect = []
        for a, n in need.items():
            actions_to_collect.extend([a] * n)
        random.shuffle(actions_to_collect)

        try:
            for i, action in enumerate(actions_to_collect, 1):
                print(f"[{i}/{total_needed}] 采集 {action}... (剩余 {total_needed - i + 1})")

                fname = f"balanced_{action}_{len(self.data):04d}.png"
                screenshot_path = self.take_screenshot(fname)

                method = ACTION_METHODS[action]
                params = method()
                self.record(action, params, str(self.save_dir / fname))

                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n[interrupt] 用户中断，保存进度...")

        finally:
            self.save()

    def save(self, filename="balanced_training_data.json"):
        filepath = self.save_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"[save] 已保存 {len(self.data)} 条到 {filepath}")

        cnt = self.current_counts()
        print("\n最终分布:")
        for a in ACTIONS:
            print(f"  {a}: {cnt.get(a, 0)}")


def main():
    print("=== 增量均衡采集 ===")
    print("将控制鼠标/键盘，请确保无重要操作")
    time.sleep(3)

    collector = BalancedDataCollector()
    json_path = collector.save_dir / "balanced_training_data.json"
    collector.load_existing(json_path)

    # 目标：每类 84 条，共 504 条
    collector.collect_to_target(target_per_action=84)


if __name__ == "__main__":
    main()