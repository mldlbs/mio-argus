import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from model_computer_use import SimpleComputerUseModel

class ComputerUseDataset(Dataset):
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
            'wait': 4
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图像
        image_path = item['screenshot']
        if not os.path.exists(image_path):
            # 如果文件不存在，创建一个占位图像
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        else:
            image = Image.open(image_path).convert('RGB')
        
        # 预处理图像
        image = self.transform(image)
        
        # 解析动作
        action_str = item['action']
        action_type = action_str.split('(')[0]  # 提取动作类型
        action_idx = self.action_to_idx.get(action_type, 4)  # 默认为wait
        
        # 简化的文本表示 (实际应该使用tokenizer)
        text_ids = torch.zeros(10, dtype=torch.long)  # 占位
        
        return {
            'image': image,
            'text_ids': text_ids,
            'action_idx': torch.tensor(action_idx, dtype=torch.long)
        }

def train_model():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SimpleComputerUseModel(num_actions=5).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 模拟训练数据 (实际应该使用真实数据)
    # 创建假数据进行验证
    fake_data = []
    for i in range(100):
        fake_data.append({
            'screenshot': f'computer_use_data/screen_{i:04d}.png',
            'action': 'click(100,200)',
            'task': f'task_{i}'
        })
    
    # 保存假数据
    os.makedirs('computer_use_data', exist_ok=True)
    with open('computer_use_data/fake_training_data.json', 'w') as f:
        json.dump(fake_data, f, indent=2)
    
    # 数据集
    dataset = ComputerUseDataset('computer_use_data/fake_training_data.json')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 训练循环
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            text_ids = batch['text_ids'].to(device)
            action_labels = batch['action_idx'].to(device)
            
            # 前向传播
            outputs = model(images, text_ids)
            loss = criterion(outputs['action_logits'], action_labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs['action_logits'].max(1)
            total += action_labels.size(0)
            correct += predicted.eq(action_labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # 打印统计
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  准确率: {accuracy:.2f}%")
    
    # 保存模型
    torch.save(model.state_dict(), 'computer_use_model.pth')
    print("模型已保存: computer_use_model.pth")
    
    return model

if __name__ == "__main__":
    print("开始训练Computer Use Model...")
    model = train_model()
    print("训练完成!")