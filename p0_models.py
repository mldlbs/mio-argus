"""
P0 模型：共享 backbone + 指令查询 → 空间热力图。

两种动作头 **参数量完全相同**，唯一差异是动作参数化：

  RegHead : heatmap → soft-argmax → (x, y) ∈ [0,1]²    连续空间，熵无上界
  ClsHead : heatmap → adaptive_avg_pool2d(g,g) → S 类   离散空间，熵 = log S

关键架构决定：backbone **不做全局池化**，保留 HxW 空间结构，
否则坐标回归在原理上就不可能（位置信息已被抹平）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialBackbone(nn.Module):
    """保留空间结构的小 CNN。width 控容量，stages 控输出分辨率。"""

    def __init__(self, width: int = 32, in_res: int = 128, stages: int = 4):
        super().__init__()
        w = width
        layers = [nn.Conv2d(3, w, 3, stride=2, padding=1), nn.BatchNorm2d(w), nn.ReLU()]
        ch = w
        for _ in range(stages - 1):
            nch = min(ch * 2, w * 4)
            layers += [nn.Conv2d(ch, nch, 3, stride=2, padding=1),
                       nn.BatchNorm2d(nch), nn.ReLU()]
            ch = nch
        self.net = nn.Sequential(*layers)
        self.out_channels = ch
        self.out_res = max(1, in_res // (2 ** stages))

    def forward(self, x):
        return self.net(x)


class P0Net(nn.Module):
    def __init__(self, width: int = 32, head: str = "reg",
                 num_slots: int = 9, grid: int = 3,
                 num_labels: int = 12, inst_dim: int = 64,
                 in_res: int = 128, stages: int = 4):
        super().__init__()
        assert head in ("reg", "cls")
        self.head = head
        self.grid = grid
        self.num_slots = num_slots

        self.backbone = SpatialBackbone(width, in_res, stages)
        C = self.backbone.out_channels
        res = self.backbone.out_res

        self.inst_emb = nn.Embedding(num_labels, inst_dim)
        self.query = nn.Linear(inst_dim, C)

        g = torch.linspace(0.5 / res, 1 - 0.5 / res, res)
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        self.register_buffer("grid_x", xx.reshape(-1))
        self.register_buffer("grid_y", yy.reshape(-1))

    def heat(self, images, inst_onehot):
        """指令×空间 的点积热力图。"""
        feat = self.backbone(images)
        B, C, h, w = feat.shape
        q = self.query(self.inst_emb(inst_onehot.argmax(dim=-1)))
        return (feat * q[..., None, None]).sum(1) / math.sqrt(C)   # (B, h, w)

    def decode(self, heat):
        B = heat.shape[0]
        p = F.softmax(heat.reshape(B, -1), dim=-1)
        x = (p * self.grid_x).sum(-1, keepdim=True)
        y = (p * self.grid_y).sum(-1, keepdim=True)
        return torch.cat([x, y], dim=-1)

    def forward(self, images, inst_onehot):
        heat = self.heat(images, inst_onehot)
        if self.head == "reg":
            return self.decode(heat)
        B = heat.shape[0]
        return F.adaptive_avg_pool2d(
            heat.unsqueeze(1), (self.grid, self.grid)).reshape(B, -1)

    @property
    def out_res(self) -> int:
        return self.backbone.out_res

    def gauss_target(self, xy, sigma: float = 1.0) -> torch.Tensor:
        """构造以 xy 为中心的高斯目标热力图 (B, res, res)，用于空间 CE 监督。"""
        B = xy.shape[0]
        res = self.out_res
        cx = (xy[:, 0] * res - 0.5).view(B, 1, 1)
        cy = (xy[:, 1] * res - 0.5).view(B, 1, 1)
        gx = torch.arange(res, device=xy.device, dtype=xy.dtype).view(1, 1, res)
        gy = torch.arange(res, device=xy.device, dtype=xy.dtype).view(1, res, 1)
        d2 = (gx - cx) ** 2 + (gy - cy) ** 2
        g = torch.exp(-d2 / (2 * sigma ** 2))
        g = g / g.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return g

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    for w in (8, 16, 32, 64):
        a = P0Net(width=w, head="reg")
        b = P0Net(width=w, head="cls")
        print(f"width={w:>3}  reg={a.n_params():>9,}  cls={b.n_params():>9,}  "
              f"identical={a.n_params() == b.n_params()}")
