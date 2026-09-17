import pyautogui
import json
import time
import os
import random
from datetime import datetime

# --- 动作执行函数 (模块级，不需要 self) ---
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
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.data = []
        self.action_count = 0
        
    def take_screenshot(self, filename):
        screenshot = pyautogui.screenshot()
        filepath = os.path.join(self.save_dir, filename)
        screenshot.save(filepath)
        return os.path.join(self.save_dir, filename)
    
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
    
    def collect_balanced(self, target_per_action=20):
        print(f"开始均衡采集: 每类 {target_per_action} 个样本...")
        
        actions = list(ACTION_METHODS.keys()) * target_per_action
        random.shuffle(actions)
        
        for i, action in enumerate(actions):
            print(f"[{i+1}/{len(actions)}] 执行 {action}...")
            
            fname = f"balanced_{action}_{self.action_count:04d}.png"
            screenshot_path = self.take_screenshot(fname)
            
            method = ACTION_METHODS[action]
            params = method()
            instruction = _make_instruction(action, params)
            
            self.record(action, params, os.path.join(self.save_dir, fname))
            
            time.sleep(0.5)
    
    def save(self, filename="balanced_training_data.json"):
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"保存完成: {filepath}, 共 {len(self.data)} 样本")
        return filepath


def main():
    print("开始均衡真实数据采集...")
    print("注意: 将控制鼠标/键盘，请确保无重要操作")
    time.sleep(3)
    
    collector = BalancedDataCollector()
    collector.collect_balanced(target_per_action=15)
    collector.save()
    
    from collections import Counter
    cnt = Counter(item["action"] for item in collector.data)
    print("\n动作分布:")
    for action, cnt in cnt.items():
        print(f"  {action}: {cnt}")

if __name__ == "__main__":
    main()