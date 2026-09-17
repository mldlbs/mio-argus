import torch
from PIL import Image
import os
import re
from model_computer_use import SimpleComputerUseModel

def test_advanced_model():
    """测试高级模型"""
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = SimpleComputerUseModel(num_actions=6).to(device)
    model.load_state_dict(torch.load('computer_use_model_advanced.pth', map_location=device))
    model.eval()
    print("模型加载完成")
    
    # 动作映射
    action_map = {
        0: "click",
        1: "type", 
        2: "scroll",
        3: "move",
        4: "hotkey",
        5: "press"
    }
    
    # 测试图像
    test_images = [
        "computer_use_data_advanced/basic_0000.png",
        "computer_use_data_advanced/basic_0001.png",
        "computer_use_data_advanced/keyboard_0000.png",
        "computer_use_data_advanced/combined_0000.png"
    ]
    
    # 测试每个图像
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n测试图像: {image_path}")
            
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))
            image_tensor = torch.tensor(list(image.getdata())).float() / 255.0
            image_tensor = image_tensor.view(3, 224, 224).unsqueeze(0).to(device)
            
            # 文本输入 (占位)
            text_ids = torch.zeros(1, 10, dtype=torch.long).to(device)
            
            # 预测
            with torch.no_grad():
                outputs = model(image_tensor, text_ids)
                
                # 动作预测
                action_logits = outputs['action_logits']
                action_probs = torch.softmax(action_logits, dim=-1)
                action_id = torch.argmax(action_probs, dim=-1).item()
                confidence = action_probs[0, action_id].item()
                action = action_map[action_id]
                
                # 坐标预测
                coords = outputs['coord_logits'][0].cpu().numpy()
                x = int(coords[0] * 1920)  # 反归一化
                y = int(coords[1] * 1080)
                
                # 滚动方向预测
                scroll_probs = torch.softmax(outputs['scroll_logits'], dim=-1)
                scroll_up = scroll_probs[0, 1].item() > 0.5
                
                print(f"  预测动作: {action}")
                print(f"  置信度: {confidence:.4f}")
                print(f"  预测坐标: ({x}, {y})")
                print(f"  滚动方向: {'上' if scroll_up else '下'}")
                print(f"  动作概率: {action_probs[0].tolist()}")
        else:
            print(f"图像不存在: {image_path}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_advanced_model()