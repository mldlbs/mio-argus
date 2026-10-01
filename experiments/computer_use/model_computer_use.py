import torch
import torch.nn as nn

try:
    from transformers import ViTModel, GPT2LMHeadModel, GPT2Config, ViTConfig
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

class ComputerUseModel(nn.Module):
    def __init__(self, num_actions=5):
        super().__init__()
        if not _HAS_TRANSFORMERS:
            raise ImportError("ComputerUseModel 需要 transformers，请先 pip install transformers")
        
        # 视觉编码器 (ViT)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vit_dim = 768  # ViT输出维度
        
        # 语言模型 (GPT-2)
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_dim = 768  # GPT-2隐藏维度
        
        # 动作映射
        self.action_map = {
            0: "click",
            1: "type", 
            2: "scroll",
            3: "move",
            4: "wait"
        }
        
        # 融合层：将视觉和文本特征融合
        self.fusion = nn.Linear(self.vit_dim + self.gpt2_dim, self.gpt2_dim)
        
        # 动作预测头
        self.action_head = nn.Linear(self.gpt2_dim, num_actions)
        
        # 坐标预测头 (用于click和move动作)
        self.coord_head = nn.Linear(self.gpt2_dim, 2)  # x, y坐标
        
    def forward(self, images, input_ids, attention_mask=None):
        # 视觉编码
        visual_outputs = self.vit(images)
        visual_features = visual_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # 文本编码
        text_outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, -1, :]  # 最后一个token
        
        # 特征融合
        combined = torch.cat([visual_features, text_features], dim=-1)
        fused = self.fusion(combined)
        
        # 动作预测
        action_logits = self.action_head(fused)
        
        # 坐标预测
        coord_logits = self.coord_head(fused)
        
        return {
            'action_logits': action_logits,
            'coord_logits': coord_logits
        }
    
    def predict(self, image, text):
        """预测动作"""
        self.eval()
        with torch.no_grad():
            # 这里需要预处理图像和文本
            # 实际使用时需要正确的预处理
            outputs = self.forward(image, text)
            
            action_id = torch.argmax(outputs['action_logits'], dim=-1).item()
            coords = torch.sigmoid(outputs['coord_logits'])  # 归一化到0-1
            
            action = self.action_map[action_id]
            x, y = coords[0].tolist()
            
            return {
                'action': action,
                'x': x,
                'y': y,
                'raw': outputs
            }

class SimpleComputerUseModel(nn.Module):
    """简化版模型，用于快速验证"""
    def __init__(self, num_actions=6, vocab_size=50257):
        super().__init__()
        
        # 简化的视觉编码器
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(64, 512)  # 64个特征
        )
        
        # 简化的文本编码器
        self.text_encoder = nn.Embedding(vocab_size, 128)
        
        # 动作预测 (6个动作: click, type, scroll, move, hotkey, press)
        self.action_head = nn.Linear(512 + 128, num_actions)
        
        # 坐标预测头 (用于click和move动作)
        self.coord_head = nn.Linear(512 + 128, 2)  # x, y坐标
        
        # 滚动方向预测 (用于scroll动作)
        self.scroll_head = nn.Linear(512 + 128, 2)  # up, down
        
    def forward(self, images, text_ids):
        # 视觉特征
        visual_features = self.visual_encoder(images)
        
        # 文本特征 (平均池化)
        text_features = self.text_encoder(text_ids).mean(dim=1)
        
        # 融合
        combined = torch.cat([visual_features, text_features], dim=-1)
        
        # 动作预测
        action_logits = self.action_head(combined)
        
        # 坐标预测 (归一化到0-1)
        coord_logits = torch.sigmoid(self.coord_head(combined))
        
        # 滚动方向预测
        scroll_logits = self.scroll_head(combined)
        
        return {
            'action_logits': action_logits,
            'coord_logits': coord_logits,
            'scroll_logits': scroll_logits
        }