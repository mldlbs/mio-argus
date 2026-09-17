"""
扩充真实数据采集 - 目标 500+ 样本
支持分类别均衡采集、指令条件化、断点续传
"""
import pyautogui
import json
import time
import os
import random
from datetime import datetime
from collections import Counter

class ExpandedDataCollector:
    def __init__(self, save_dir="computer_use_data_expanded", target_per_action=100):
        self.save_dir = save_dir
        self.target_per_action = target_per_action
        os.makedirs(save_dir, exist_ok=True)
        self.data_file = os.path.join(save_dir, "expanded_training_data.json")
        self.data = []
        self.action_count = 0
        self._load_existing()
        
    def _load_existing(self):
        """加载已有数据，支持断点续传"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.action_count = len(self.data)
            print(f"[resume] 加载已有数据: {self.action_count} 样本")
            
    def _make_instruction(self, action, params):
        templates = {
            "click": [f"click at {{x}} {{y}}", "click the button at {{x}} {{y}}", "click on {{x}} {{y}}"],
            "type": [f"type {{text}} in the input", "enter {{text}} here", "input {{text}}"],
            "scroll": [f"scroll {{direction}} the page", "scroll {{direction}} down", "scroll {{direction}} up"],
            "move": [f"move mouse to {{x}} {{y}}", "move cursor to {{x}} {{y}}", "move pointer to {{x}} {{y}}"],
            "hotkey": [f"press {{keys}}", "use shortcut {{keys}}", "trigger {{keys}}"],
            "press": [f"press {{key}}", "hit {{key}}", "tap {{key}}"],
        }
        template = random.choice(templates.get(action, ["{action}"]))
        return template.format(**params)
    
    def _collect_click(self):
        x = random.randint(100, 1800)
        y = random.randint(100, 950)
        pyautogui.click(x, y)
        return {"x": x, "y": y}
    
    def _collect_type(self):
        text = random.choice(["hello world", "test input", "computer use model", "data collection", "training data", "machine learning"])
        pyautogui.typewrite(text, interval=0.01)
        return {"text": text}
    
    def _collect_scroll(self):
        direction = random.choice(["up", "down"])
        amount = random.randint(1, 5)
        pyautogui.scroll(amount if direction == "up" else -amount)
        return {"direction": direction, "amount": amount}
    
    def _collect_move(self):
        x = random.randint(100, 1800)
        y = random.randint(100, 950)
        pyautogui.moveTo(x, y)
        return {"x": x, "y": y}
    
    def _collect_hotkey(self):
        keys = random.choice([["ctrl", "c"], ["ctrl", "v"], ["ctrl", "x"], ["alt", "tab"], ["ctrl", "s"], ["ctrl", "z"]])
        pyautogui.hotkey(*keys)
        return {"keys": keys}
    
    def _collect_press(self):
        key = random.choice(["enter", "tab", "escape", "space", "backspace", "delete"])
        pyautogui.press(key)
        return {"key": key}
    
    ACTION_METHODS = {
        "click": "_collect_click",
        "type": "_collect_type",
        "scroll": "_collect_scroll",
        "move": "_collect_move",
        "hotkey": "_collect_hotkey",
        "press": "_collect_press",
    }
    
    def _make_instruction(self, action, params):
        templates = {
            "click": ["click at {x} {y}", "click the button at {x} {y}", "click on {x} {y}"],
            "type": ["type {text} in the input", "enter {text} here", "input {text}"],
            "scroll": ["scroll {direction} the page", "scroll {direction} down", "scroll {direction} up"],
            "move": ["move mouse to {x} {y}", "move cursor to {x} {y}", "move pointer to {x} {y}"],
            "hotkey": ["press {keys}", "use shortcut {keys}", "trigger {keys}"],
            "press": ["press {key}", "hit {key}", "tap {key}"],
        }
        template = random.choice(templates.get(action, ["{action}"]))
        # 格式化 keys 列表
        if "keys" in params:
            params = params.copy()
            params["keys"] = " + ".join(params["keys"])
        return random.choice(templates.get(action, ["{action}"])).format(**params)
    
    def collect_balanced(self, target_per_action=100):
        print(f"开始均衡采集: 每类 {target_per_action} 个样本 (共 {6 * target_per_action})...")
        print("注意: 将控制鼠标/键盘，请确保无重要操作")
        time.sleep(3)
        
        # 统计已有各类别数量
        action_counts = Counter(item["action"] for item in self.data)
        
        # 计算每类还需要采集的数量
        actions = list(self.ACTION_METHODS.keys())
        for action in actions:
            current = action_counts.get(action, 0)
            needed = max(0, target_per_action - current)
            if needed > 0:
                print(f"  {action}: 已有 {current}, 需补充 {needed}")
        
        # 生成采集计划
        plan = []
        for action in actions:
            current = action_counts.get(action, 0)
            needed = max(0, target_per_action - current)
            plan.extend([action] * needed)
        
        # 打乱顺序
        random.shuffle(plan)
        
        print(f"\n开始采集 {len(plan)} 个样本...")
        
        for i, action in enumerate(plan):
            print(f"[{i+1}/{len(plan)}] 执行 {action}...")
            
            fname = f"expanded_{self.action_count:05d}.png"
            screenshot_path = self.take_screenshot(fname)
            
            method = getattr(self, self.ACTION_METHODS[action])
            params = method()
            instruction = self._make_instruction(action, params)
            
            record = {
                "timestamp": datetime.now().isoformat(),
                "screenshot": screenshot_path,
                "instruction": instruction,
                "action": action,
                "params": params
            }
            self.data.append(record)
            self.action_count += 1
            
            # 每 10 个样本保存一次
            if self.action_count % 10 == 0:
                self.save()
            
            time.sleep(0.5)
        
        self.save()
        print(f"\n采集完成! 总计 {len(self.data)} 样本")
        self._print_stats()
    
    def take_screenshot(self, filename):
        screenshot = pyautogui.screenshot()
        filepath = os.path.join(self.save_dir, filename)
        screenshot.save(filepath)
        return filepath
    
    def save(self, filename=None):
        filepath = filename or os.path.join(self.save_dir, "expanded_training_data.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"[save] 已保存 {len(self.data)} 样本到 {filepath}")
    
    def _print_stats(self):
        cnt = Counter(item["action"] for item in self.data)
        print("\n动作分布:")
        for action, count in sorted(cnt.items()):
            print(f"  {action}: {count}")

def main():
    print("=" * 50)
    print("Computer Use 真实数据扩充采集工具")
    print("=" * 50)
    
    collector = ExpandedDataCollector(target_per_action=100)  # 6类 × 100 = 600
    collector.collect_balanced(target_per_action=100)

if __name__ == "__main__":
    main()