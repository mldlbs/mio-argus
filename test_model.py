import torch
from PIL import Image
import os
from model_computer_use import SimpleComputerUseModel

def test_model():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = SimpleComputerUseModel(num_actions=5).to(device)
    model.load_state_dict(torch.load('computer_use_model.pth', map_location=device))
    model.eval()
    print("模型加载完成")
    
    # 动作映射
    action_map = {
        0: "click",
        1: "type", 
        2: "scroll",
        3: "move",
        4: "wait"
    }
    
    # 测试图像
    test_images = [
        "computer_use_data/screen_0000.png",
        "computer_use_data/screen_0001.png",
        "computer_use_data/screen_0002.png",
        "computer_use_data/screen_0003.png"
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
                action_logits = outputs['action_logits']
                action_probs = torch.softmax(action_logits, dim=-1)
                action_id = torch.argmax(action_probs, dim=-1).item()
                confidence = action_probs[0, action_id].item()
                
                action = action_map[action_id]
                
                print(f"  预测动作: {action}")
                print(f"  置信度: {confidence:.4f}")
                print(f"  动作概率: {action_probs[0].tolist()}")
        else:
            print(f"图像不存在: {image_path}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_model()