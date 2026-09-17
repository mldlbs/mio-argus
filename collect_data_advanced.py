import pyautogui
import json
import time
import os
import random
from datetime import datetime

class AdvancedDataCollector:
    def __init__(self, save_dir="computer_use_data_advanced"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.data = []
        self.action_count = 0
        
    def take_screenshot(self, filename):
        """截图"""
        screenshot = pyautogui.screenshot()
        filepath = os.path.join(self.save_dir, filename)
        screenshot.save(filepath)
        return filepath
    
    def record_action(self, screenshot_path, action, task, metadata=None):
        """记录动作"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "screenshot": screenshot_path,
            "action": action,
            "task": task,
            "metadata": metadata or {}
        }
        self.data.append(record)
        self.action_count += 1
        return record
    
    def get_random_position(self):
        """获取随机屏幕位置"""
        screen_width, screen_height = pyautogui.size()
        x = random.randint(100, screen_width - 100)
        y = random.randint(100, screen_height - 100)
        return x, y
    
    def collect_basic_actions(self, num_samples=20):
        """收集基本动作数据"""
        print("收集基本动作数据...")
        
        for i in range(num_samples):
            # 截图
            screenshot_file = f"basic_{i:04d}.png"
            screenshot_path = self.take_screenshot(screenshot_file)
            
            # 随机选择动作
            action_type = random.choice(["click", "type", "scroll", "move"])
            
            if action_type == "click":
                x, y = self.get_random_position()
                pyautogui.click(x, y)
                action = f"click({x},{y})"
                metadata = {"x": x, "y": y}
                
            elif action_type == "type":
                text = random.choice([
                    "hello", "test", "computer use model", 
                    "python", "training", "data"
                ])
                pyautogui.typewrite(text)
                action = f"type('{text}')"
                metadata = {"text": text}
                
            elif action_type == "scroll":
                direction = random.choice(["up", "down"])
                amount = random.randint(1, 5)
                if direction == "up":
                    pyautogui.scroll(amount)
                else:
                    pyautogui.scroll(-amount)
                action = f"scroll({direction},{amount})"
                metadata = {"direction": direction, "amount": amount}
                
            elif action_type == "move":
                x, y = self.get_random_position()
                pyautogui.moveTo(x, y)
                action = f"move({x},{y})"
                metadata = {"x": x, "y": y}
            
            # 记录
            self.record_action(screenshot_path, action, f"basic_{action_type}", metadata)
            time.sleep(0.5)
    
    def collect_keyboard_actions(self, num_samples=20):
        """收集键盘动作数据"""
        print("收集键盘动作数据...")
        
        for i in range(num_samples):
            # 截图
            screenshot_file = f"keyboard_{i:04d}.png"
            screenshot_path = self.take_screenshot(screenshot_file)
            
            # 随机选择键盘动作
            action_type = random.choice(["hotkey", "press", "type_long"])
            
            if action_type == "hotkey":
                keys = random.choice([
                    ['ctrl', 'c'], ['ctrl', 'v'], ['ctrl', 'x'],
                    ['alt', 'tab'], ['win', 'r'], ['ctrl', 's']
                ])
                pyautogui.hotkey(*keys)
                action = f"hotkey({','.join(keys)})"
                metadata = {"keys": keys}
                
            elif action_type == "press":
                key = random.choice([
                    'enter', 'tab', 'escape', 'space',
                    'backspace', 'delete', 'up', 'down'
                ])
                pyautogui.press(key)
                action = f"press({key})"
                metadata = {"key": key}
                
            elif action_type == "type_long":
                text = "This is a longer text for testing the computer use model capabilities."
                pyautogui.typewrite(text, interval=0.01)
                action = f"type('{text}')"
                metadata = {"text": text}
            
            # 记录
            self.record_action(screenshot_path, action, f"keyboard_{action_type}", metadata)
            time.sleep(0.5)
    
    def collect_combined_actions(self, num_samples=20):
        """收集组合动作数据"""
        print("收集组合动作数据...")
        
        for i in range(num_samples):
            # 截图
            screenshot_file = f"combined_{i:04d}.png"
            screenshot_path = self.take_screenshot(screenshot_file)
            
            # 组合动作
            actions = []
            
            # 先移动
            x, y = self.get_random_position()
            pyautogui.moveTo(x, y)
            actions.append(f"move({x},{y})")
            
            # 然后点击
            pyautogui.click(x, y)
            actions.append(f"click({x},{y})")
            
            # 然后输入
            text = random.choice(["test", "hello", "data"])
            pyautogui.typewrite(text)
            actions.append(f"type('{text}')")
            
            action = "; ".join(actions)
            metadata = {"x": x, "y": y, "text": text}
            
            # 记录
            self.record_action(screenshot_path, action, "combined_action", metadata)
            time.sleep(1)
    
    def save_data(self, filename="advanced_training_data.json"):
        """保存数据"""
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"数据已保存到: {filepath}")
        print(f"总动作数: {self.action_count}")
        return filepath
    
    def collect_all(self, samples_per_category=20):
        """收集所有类型数据"""
        print("开始高级数据收集...")
        
        self.collect_basic_actions(samples_per_category)
        self.collect_keyboard_actions(samples_per_category)
        self.collect_combined_actions(samples_per_category)
        
        self.save_data()
        print(f"数据收集完成! 共收集 {self.action_count} 个动作")

def main():
    """主函数"""
    collector = AdvancedDataCollector()
    
    # 收集数据 (每个类别20个样本，共60个)
    collector.collect_all(samples_per_category=20)
    
    print("\n数据收集统计:")
    print(f"总样本数: {len(collector.data)}")
    
    # 统计动作类型
    action_types = {}
    for item in collector.data:
        action = item['action']
        action_type = action.split('(')[0]
        action_types[action_type] = action_types.get(action_type, 0) + 1
    
    print("动作类型分布:")
    for action_type, count in action_types.items():
        print(f"  {action_type}: {count}")

if __name__ == "__main__":
    main()