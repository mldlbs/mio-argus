"""
Computer Use Model v3: Full ViT + GPT-2 Architecture
- ViT 视觉编码器
- GPT-2 语言模型 (指令+动作序列建模)
- Cross-Attention 融合
- Huber Loss 坐标回归
- 动作分类 + 坐标回归 + 滚动方向
"""
import os
if not os.environ.get('TRANSFORMERS_OFFLINE'):
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from transformers import ViTModel, GPT2Model, AutoImageProcessor, AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

ACTION_MAP = {
    0: "click", 1: "type", 2: "scroll", 3: "move",
    4: "hotkey", 5: "press"
}
ACTION_TO_IDX = {v: k for k, v in ACTION_MAP.items()}
NUM_ACTIONS = len(ACTION_MAP)


class LoRALinear(nn.Linear):
    """LoRA 适配器包装 Linear 层"""

    def __init__(self, in_features, out_features, rank=16, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = rank
        self.lora_scaling = 1.0 / (rank ** 0.5)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + (
            x @ self.lora_A.t() @ self.lora_B.t()
        ) * self.lora_scaling


def apply_lora(model, target_names=('q_proj', 'k_proj', 'v_proj', 'out_proj'), rank=16):
    """Replace matching linear modules with LoRALinear modules."""
    if rank <= 0:
        raise ValueError("rank must be positive")

    target_names = tuple(target_names)
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not any(name == target or name.endswith('.' + target) for target in target_names):
            continue

        replacement = LoRALinear(
            module.in_features,
            module.out_features,
            rank=rank,
            bias=module.bias is not None,
        )
        replacement.load_state_dict(module.state_dict(), strict=False)
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, replacement)


class CrossAttentionFusion(nn.Module):
    """Cross-Attention: Query=GPT2, Key/Value=ViT"""
    def __init__(self, d_model=768, n_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, text_feat, visual_feat):
        attn_out, _ = self.cross_attn(
            query=text_feat,
            key=visual_feat,
            value=visual_feat
        )
        x = self.norm1(text_feat + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class ComputerUseModelV3(nn.Module):
    def __init__(
        self,
        num_actions=6,
        freeze_vit=True,
        freeze_gpt2=True,
        coord_loss_weight=1.0,
        scroll_loss_weight=1.0,
        lora_rank=16,
    ):
        super().__init__()
        if not _HAS_TRANSFORMERS:
            raise ImportError("需要 transformers")

        self.num_actions = num_actions
        self.freeze_vit = freeze_vit
        self.freeze_gpt2 = freeze_gpt2
        self.coord_loss_weight = coord_loss_weight
        self.scroll_loss_weight = scroll_loss_weight
        self.lora_rank = lora_rank

        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.gpt2 = GPT2Model.from_pretrained('gpt2')

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False
        if freeze_gpt2:
            for p in self.gpt2.parameters():
                p.requires_grad = False

        self._apply_lora(self.vit)
        self._apply_lora(self.gpt2)

        self.action_emb = nn.Embedding(10, 768)
        self.fusion = CrossAttentionFusion(d_model=768, n_heads=8)
        self.out_proj = nn.Linear(768, 768)

        self.action_head = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, num_actions)
        )
        self.coord_head = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 2), nn.Sigmoid()
        )
        self.scroll_head = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 3)
        )

        self._init_weights()

    def _apply_lora(self, model):
        apply_lora(model, rank=self.lora_rank)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images, input_ids, attention_mask=None):
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.parameters())):
            vit_out = self.vit(pixel_values=images)
            visual_feat = vit_out.last_hidden_state

        gpt2_out = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = gpt2_out.last_hidden_state

        fused = self.fusion(text_feat, visual_feat)

        if attention_mask is not None:
            seq_len = attention_mask.sum(dim=1) - 1
            fused_last = torch.stack([fused[i, seq_len[i]] for i in range(fused.size(0))])
        else:
            fused_last = fused[:, -1, :]

        fused_last = self.out_proj(fused_last)

        action_logits = self.action_head(fused_last)
        coord_pred = self.coord_head(fused_last)
        scroll_logits = self.scroll_head(fused_last)

        return {
            'action_logits': action_logits,
            'coord_pred': coord_pred,
            'scroll_logits': scroll_logits,
        }

    def compute_loss(self, outputs, targets, action_weights=None):
        action_logits = outputs['action_logits']
        coord_pred = outputs['coord_pred']
        scroll_logits = outputs['scroll_logits']

        action = targets['action'].to(device=action_logits.device)
        coords = targets['coords'].to(device=action_logits.device)
        scroll = targets['scroll'].to(device=action_logits.device)
        coord_mask = targets.get(
            'coord_mask',
            torch.ones_like(coords, dtype=torch.bool),
        )
        coord_mask = coord_mask.to(device=coords.device, dtype=torch.bool)

        action_loss = F.cross_entropy(action_logits, action, weight=action_weights)

        if coord_mask.ndim == 0:
            if coord_mask.item():
                mask_values = torch.ones(coords.numel(), device=coords.device, dtype=torch.bool)
            else:
                mask_values = None
        elif coord_mask.ndim == 1:
            if coord_mask.size(0) != coords.size(0):
                raise ValueError(
                    f"per-sample coord_mask must have {coords.size(0)} entries; "
                    f"got {coord_mask.size(0)}"
                )
            mask_values = coord_mask[:, None].expand_as(coords).reshape(-1)
        elif coord_mask.shape == coords.shape:
            mask_values = coord_mask.reshape(-1)
        else:
            raise ValueError(
                f"coord_mask must be scalar, per-sample ({coords.shape[0]}), "
                f"or match coords ({coords.shape}); got {tuple(coord_mask.shape)}"
            )

        coord_values = coord_pred.reshape(-1)
        target_values = coords.reshape(-1)
        if mask_values is not None and mask_values.numel() != coord_values.size(0):
            raise ValueError("coord_mask does not cover every coordinate in coord_pred")

        if mask_values is not None and bool(mask_values.any().item()):
            coord_loss = F.huber_loss(
                coord_values[mask_values],
                target_values[mask_values],
                delta=0.1,
                reduction='mean',
            )
        else:
            coord_loss = torch.zeros((), device=action.device)
        coord_loss = coord_loss * self.coord_loss_weight

        scroll_loss = F.cross_entropy(scroll_logits, scroll)
        scroll_loss = scroll_loss * self.scroll_loss_weight

        return action_loss + coord_loss + scroll_loss

    @torch.no_grad()
    def predict(self, images, input_ids, attention_mask=None):
        self.eval()
        outputs = self.forward(images, input_ids, attention_mask)
        action_id = outputs['action_logits'].argmax(dim=-1).item()
        action = ACTION_MAP.get(action_id, 'wait')
        coord = outputs['coord_pred'][0].cpu().flatten().tolist()
        scroll_id = outputs['scroll_logits'].argmax(dim=-1).item()
        scroll = ['up', 'down', 'none'][scroll_id]

        return {
            'action': action,
            'coord': coord,
            'scroll': scroll,
        }


class ComputerUseModelV3Small(nn.Module):
    def __init__(self, num_actions=6, lora_rank=16):
        super().__init__()
        if not _HAS_TRANSFORMERS:
            raise ImportError("需要 transformers")

        self.num_actions = num_actions
        self.lora_rank = lora_rank

        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.gpt2 = GPT2Model.from_pretrained('gpt2')

        for p in self.vit.parameters():
            p.requires_grad = False
        for p in self.gpt2.parameters():
            p.requires_grad = False

        self._apply_lora(self.vit)
        self._apply_lora(self.gpt2)

        self.fusion = CrossAttentionFusion()

        self.action_head = nn.Linear(768, num_actions)
        self.coord_head = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 2), nn.Sigmoid()
        )
        self.scroll_head = nn.Linear(768, 3)

    def _apply_lora(self, model):
        apply_lora(model, rank=self.lora_rank)

    def forward(self, images, input_ids, attention_mask=None):
        vit_out = self.vit(pixel_values=images)
        visual_feat = vit_out.last_hidden_state

        gpt2_out = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = gpt2_out.last_hidden_state

        fused = self.fusion(text_feat, visual_feat)

        fused_last = fused[:, -1, :]

        return {
            'action_logits': self.action_head(fused_last),
            'coord_pred': self.coord_head(fused_last),
            'scroll_logits': self.scroll_head(fused_last),
        }


# 兼容性: 保留 SimpleComputerUseModel
class SimpleComputerUseModel(nn.Module):
    def __init__(self, num_actions=6, vocab_size=50257):
        super().__init__()

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 512)
        )

        self.text_encoder = nn.Embedding(vocab_size, 128)

        self.action_head = nn.Linear(512 + 128, 6)
        self.coord_head = nn.Linear(512 + 128, 2)
        self.scroll_head = nn.Linear(512 + 128, 2)

    def forward(self, images, text_ids):
        visual_features = self.visual_encoder(images)
        text_features = self.text_encoder(text_ids).mean(dim=1)
        combined = torch.cat([visual_features, text_features], dim=-1)

        return {
            'action_logits': self.action_head(combined),
            'coord_logits': torch.sigmoid(self.coord_head(combined)),
            'scroll_logits': self.scroll_head(combined)
        }


# 预处理工具
def get_image_processor():
    if _HAS_TRANSFORMERS:
        return AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    return None


def get_tokenizer():
    if _HAS_TRANSFORMERS:
        tok = AutoTokenizer.from_pretrained('gpt2')
        tok.pad_token = tok.eos_token
        return tok
    return None
