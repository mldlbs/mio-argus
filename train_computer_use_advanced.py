import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
import re
from model_computer_use import SimpleComputerUseModel

class AdvancedComputerUseDataset(Dataset):
    def __init__(self, data_file, transform=None):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 动作到索引的映射
        self.action_to_idx = {
            'click': 0,
            'type': 1,
            'scroll': 2,
            'move': 3,
            'hotkey': 4,
            'press': 5
        }
        
    def __len__(self):
        return len(self.data)
    
    def parse_action(self, action_str):
        """解析动作字符串"""
        # 提取动作类型
        action_type = action_str.split('(')[0]
        
        # 提取参数
        params = {}
        if action_type in ['click', 'move']:
            # 提取坐标: click(100,200) 或 move(100,200)
            match = re.search(r'\((\d+),(\d+)\)', action_str)
            if match:
                params['x'] = int(match.group(1))
                params['y'] = int(match.group(2))
                
        elif action_type == 'scroll':
            # 提取滚动参数: scroll(down,2)
            match = re.search(r'\((\w+),(\d+)\)', action_str)
            if match:
                params['direction'] = match.group(1)
                params['amount'] = int(match.group(2))
                
        elif action_type == 'type':
            # 提取文本: type('hello')
            match = re.search(r"'(.*?)'", action_str)
            if match:
                params['text'] = match.group(1)
                
        elif action_type == 'hotkey':
            # 提取按键: hotkey(ctrl,c)
            match = re.search(r'\((.*?)\)', action_str)
            if match:
                params['keys'] = match.group(1).split(',')
                
        elif action_type == 'press':
            # 提取按键: press(enter)
            match = re.search(r'\((.*?)\)', action_str)
            if match:
                params['key'] = match.group(1)
        
        return action_type, params
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图像
        image_path = item['screenshot']
        if not os.path.exists(image_path):
            # 如果文件不存在，创建一个占位图像
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except:
                image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        # 预处理图像
        image = self.transform(image)
        
        # 解析动作
        action_str = item['action']
        action_type, params = self.parse_action(action_str)
        action_idx = self.action_to_idx.get(action_type, 5)  # 默认为press
        
        # 提取坐标 (如果有)
        x = params.get('x', 0) / 1920.0  # 归一化到0-1 (假设屏幕宽度1920)
        y = params.get('y', 0) / 1080.0  # 归一化到0-1 (假设屏幕高度1080)
        
        # 提取滚动方向
        scroll_up = 1 if params.get('direction') == 'up' else 0
        
        # 简化的文本表示 (实际应该使用tokenizer)
        text_ids = torch.zeros(10, dtype=torch.long)  # 占位
        
        return {
            'image': image,
            'text_ids': text_ids,
            'action_idx': torch.tensor(action_idx, dtype=torch.long),
            'coords': torch.tensor([x, y], dtype=torch.float),
            'scroll_up': torch.tensor(scroll_up, dtype=torch.long)
        }

def train_advanced_model():
    """训练高级模型"""
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SimpleComputerUseModel(num_actions=6).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数
    action_criterion = nn.CrossEntropyLoss()
    coord_criterion = nn.MSELoss()
    scroll_criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 数据集
    dataset = AdvancedComputerUseDataset('computer_use_data_advanced/advanced_training_data.json')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"数据集大小: {len(dataset)}")
    
    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        action_correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            text_ids = batch['text_ids'].to(device)
            action_labels = batch['action_idx'].to(device)
            coord_labels = batch['coords'].to(device)
            scroll_labels = batch['scroll_up'].to(device)
            
            # 前向传播
            outputs = model(images, text_ids)
            
            # 计算损失
            action_loss = action_criterion(outputs['action_logits'], action_labels)
            coord_loss = coord_criterion(outputs['coord_logits'], coord_labels)
            scroll_loss = scroll_criterion(outputs['scroll_logits'], scroll_labels)
            
            # 总损失 (加权)
            loss = action_loss + 0.5 * coord_loss + 0.5 * scroll_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs['action_logits'].max(1)
            total += action_labels.size(0)
            action_correct += predicted.eq(action_labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # 打印统计
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * action_correct / total
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  动作准确率: {accuracy:.2f}%")
    
    # 保存模型
    torch.save(model.state_dict(), 'computer_use_model_advanced.pth')
    print("模型已保存: computer_use_model_advanced.pth")
    
    return model

if __name__ == "__main__":
    print("开始训练高级Computer Use Model...")
    model = train_advanced_model()
    print("训练完成!")