import pyautogui
import json
import time
import os
from datetime import datetime

class DataCollector:
    def __init__(self, save_dir="computer_use_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.data = []
        
    def take_screenshot(self, filename):
        screenshot = pyautogui.screenshot()
        filepath = os.path.join(self.save_dir, filename)
        screenshot.save(filepath)
        return filepath
    
    def record_action(self, screenshot_path, action, task):
        record = {
            "timestamp": datetime.now().isoformat(),
            "screenshot": screenshot_path,
            "action": action,
            "task": task
        }
        self.data.append(record)
        return record
    
    def collect_automated_data(self, tasks):
        """自动化收集数据"""
        for i, task in enumerate(tasks):
            print(f"执行任务: {task}")
            
            # 截图
            screenshot_file = f"screen_{i:04d}.png"
            screenshot_path = self.take_screenshot(screenshot_file)
            
            # 根据任务执行操作
            if task == "打开记事本":
                pyautogui.hotkey('win', 'r')
                time.sleep(0.5)
                pyautogui.typewrite('notepad')
                pyautogui.press('enter')
                action = "hotkey(win,r); type('notepad'); press(enter)"
                
            elif task == "输入文字":
                pyautogui.typewrite('Hello, this is a test from computer use model!')
                action = "type('Hello, this is a test from computer use model!')"
                
            elif task == "保存文件":
                pyautogui.hotkey('ctrl', 's')
                time.sleep(1)
                pyautogui.typewrite('test_file.txt')
                pyautogui.press('enter')
                action = "hotkey(ctrl,s); type('test_file.txt'); press(enter)"
                
            elif task == "关闭窗口":
                pyautogui.hotkey('alt', 'f4')
                action = "hotkey(alt,f4)"
            
            # 记录
            self.record_action(screenshot_path, action, task)
            time.sleep(1)
    
    def save_data(self, filename="training_data.json"):
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"数据已保存到: {filepath}")
        return filepath

# 预定义任务列表
AUTOMATED_TASKS = [
    "打开记事本",
    "输入文字", 
    "保存文件",
    "关闭窗口"
]

def main():
    print("开始自动化数据收集...")
    collector = DataCollector()
    
    # 执行自动化任务
    collector.collect_automated_data(AUTOMATED_TASKS)
    
    # 保存数据
    collector.save_data()
    
    print(f"收集了 {len(collector.data)} 条数据")
    print("数据收集完成!")

if __name__ == "__main__":
    main()