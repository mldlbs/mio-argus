"""
W30-A：网格坐标头（替换绝对坐标回归）

动机（W30_RESULTS §4）：坐标回归在留出链上中位误差 358px、命中 0/341，
而按钮约 32px。连续回归 + 冻结主干拿不到亚 token 级精度；改成**沿每个轴做
G 个 bin 的分类**，再用软期望解码成连续坐标：

  col_logits (B,G) → p_col → x = Σ p_col · (i+0.5)/G
  row_logits (B,G) → p_row → y = Σ p_row · (i+0.5)/G

为什么是两轴分离而不是 G×G 联合分类：联合分类有 G² 类、1394 条样本远不够；
两轴各 G 类，样本利用率高，且屏幕坐标本来就可分解。

G=128 时单 bin 覆盖 2560/128=20px（横）× 1440/128=11.25px（纵），
半 bin 误差约 10px，理论上够命中 32px 按钮。

与基线的关系：`ComputerUseModelV3` 不动（其他线在用），本文件只做实验侧派生类。
新头的权重不在 v3 checkpoint 里，加载时 strict=False 会缺键，属预期。
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(ROOT / "mio"), str(ROOT)) if p not in sys.path]

from mio_argus import ComputerUseModelV3  # noqa: E402


class GridHead(nn.Module):
    def __init__(self, d_model=768, grid=128, hidden=512):
        super().__init__()
        self.grid = grid
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
        )
        self.row = nn.Linear(hidden // 2, grid)
        self.col = nn.Linear(hidden // 2, grid)

    def forward(self, x):
        h = self.trunk(x)
        row_logits, col_logits = self.row(h), self.col(h)
        # 软解码：期望坐标，可微，评测时也用它
        idx = (torch.arange(self.grid, device=x.device, dtype=x.dtype) + 0.5) / self.grid
        col = (F.softmax(col_logits, dim=-1) * idx).sum(-1)
        row = (F.softmax(row_logits, dim=-1) * idx).sum(-1)
        return row_logits, col_logits, torch.stack([col, row], dim=-1)


class SpatialCoordHead(nn.Module):
    """先对视觉 token 网格出热力图选 token，再在该 token 内回归残差。

    与 GridHead 的关键差别：输入是 (B, N, D) 的**空间 token**，不是全局池化向量，
    因此可以表达"图上第几行第几列"；GridHead 只能从全局向量里记映射。
    """

    def __init__(self, d_model=768, hidden=256):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                   nn.Linear(hidden, 1))
        self.res = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2), nn.Tanh())

    @staticmethod
    def token_centers(n, device, dtype):
        g = int(round(n ** 0.5))
        assert g * g == n, f"token 数 {n} 不是平方数"
        ys = (torch.arange(g, device=device, dtype=dtype) + 0.5) / g
        xs = (torch.arange(g, device=device, dtype=dtype) + 0.5) / g
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1), g

    def forward(self, tokens):
        centers, g = self.token_centers(tokens.size(1), tokens.device, tokens.dtype)
        logits = self.score(tokens).squeeze(-1)            # B,N
        p = F.softmax(logits, dim=-1)
        coarse = p @ centers                                # B,2
        feat = (p.unsqueeze(-1) * tokens).sum(1)            # B,D
        res = self.res(feat)                                # B,2 in [-1,1]
        xy = coarse + res * (0.5 / g)                       # 残差限制在半个 token 内
        return logits, xy


class SpatialModel(ComputerUseModelV3):
    """复用 v3 主干，坐标头改成"token 热力图 + token 内残差"。"""

    def __init__(self, num_actions=6, freeze_vit=True, freeze_gpt2=True):
        super().__init__(num_actions=num_actions, freeze_vit=freeze_vit, freeze_gpt2=freeze_gpt2)
        self.coord_head = SpatialCoordHead()

    def forward(self, images, input_ids, attention_mask=None):
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.parameters())):
            visual_feat = self.vit(pixel_values=images, interpolate_pos_encoding=True).last_hidden_state
        text_feat = self.gpt2(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        fused = self.fusion(text_feat, visual_feat)

        if attention_mask is not None:
            seq_len = attention_mask.sum(dim=1) - 1
            fused_last = torch.stack([fused[i, seq_len[i]] for i in range(fused.size(0))])
        else:
            fused_last = fused[:, -1, :]
        fused_last = self.out_proj(fused_last)

        coord_logits, coord_pred = self.coord_head(visual_feat[:, 1:])  # 去掉 CLS，只留 patch 网格
        return {
            "action_logits": self.action_head(fused_last),
            "coord_pred": coord_pred,
            "coord_logits": coord_logits,
            "scroll_logits": self.scroll_head(fused_last),
        }

    def compute_loss(self, outputs, targets, action_weights=None):
        device = outputs["action_logits"].device
        action = targets["action"].to(device)
        coords = targets["coords"].to(device)
        scroll = targets["scroll"].to(device)
        mask = targets.get("coord_mask", torch.ones_like(coords, dtype=torch.bool))
        mask = mask.to(device=device).bool()

        loss = F.cross_entropy(outputs["action_logits"], action, weight=action_weights)
        loss = loss + F.cross_entropy(outputs["scroll_logits"], scroll)

        if bool(mask.any()):
            logits = outputs["coord_logits"]
            g = int(round(logits.size(1) ** 0.5))
            cx = torch.clamp((coords[:, 0] * g).long(), 0, g - 1)
            cy = torch.clamp((coords[:, 1] * g).long(), 0, g - 1)
            token = (cy * g + cx)[mask]
            loss = loss + F.cross_entropy(logits[mask], token)
            loss = loss + F.huber_loss(outputs["coord_pred"][mask], coords[mask], delta=0.1) * 10.0
        return loss


class GridModel(ComputerUseModelV3):
    """复用 v3 的 ViT/GPT-2/fusion/action/scroll，只把 coord 头换成网格分类。"""

    def __init__(self, num_actions=6, freeze_vit=True, freeze_gpt2=True, grid=128):
        super().__init__(num_actions=num_actions, freeze_vit=freeze_vit, freeze_gpt2=freeze_gpt2)
        self.grid = grid
        self.coord_head = GridHead(d_model=768, grid=grid)

    def forward(self, images, input_ids, attention_mask=None):
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.parameters())):
            visual_feat = self.vit(pixel_values=images, interpolate_pos_encoding=True).last_hidden_state
        text_feat = self.gpt2(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        fused = self.fusion(text_feat, visual_feat)

        if attention_mask is not None:
            seq_len = attention_mask.sum(dim=1) - 1
            fused_last = torch.stack([fused[i, seq_len[i]] for i in range(fused.size(0))])
        else:
            fused_last = fused[:, -1, :]
        fused_last = self.out_proj(fused_last)

        row_logits, col_logits, coord_pred = self.coord_head(fused_last)
        return {
            "action_logits": self.action_head(fused_last),
            "coord_pred": coord_pred,
            "coord_logits": (row_logits, col_logits),
            "scroll_logits": self.scroll_head(fused_last),
        }

    def compute_loss(self, outputs, targets, action_weights=None):
        device = outputs["action_logits"].device
        action = targets["action"].to(device)
        coords = targets["coords"].to(device)
        scroll = targets["scroll"].to(device)
        mask = targets.get("coord_mask", torch.ones_like(coords, dtype=torch.bool))
        mask = mask.to(device=device).bool()

        loss = F.cross_entropy(outputs["action_logits"], action, weight=action_weights)
        loss = loss + F.cross_entropy(outputs["scroll_logits"], scroll)

        if bool(mask.any()):
            g = self.grid
            # 归一坐标 -> bin（clamp 防 1.0 越界）
            col_bin = torch.clamp((coords[:, 0] * g).long(), 0, g - 1)[mask]
            row_bin = torch.clamp((coords[:, 1] * g).long(), 0, g - 1)[mask]
            row_logits, col_logits = outputs["coord_logits"]
            loss = loss + F.cross_entropy(row_logits[mask], row_bin)
            loss = loss + F.cross_entropy(col_logits[mask], col_bin)
        return loss
