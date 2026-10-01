"""验证 VerifierNet 与 P0Net(cls) 在共享权重时是否给出相同排序。"""
import numpy as np
import torch

from p0_env import LABELS, instruction_onehot
from p0_models import P0Net
from p0_h2 import load_or_build, cfg_for, to_t
from p1_horizon import GRID
from p2_verifier import VerifierNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = cfg_for(GRID, fill=0.40)
data = load_or_build(500, 999, cfg, "m")

pi = P0Net(width=32, head="cls", num_slots=GRID * GRID, grid=GRID,
           num_labels=len(LABELS), in_res=cfg.img_size, stages=3).to(device)
v = VerifierNet(width=32, grid=GRID, in_res=cfg.img_size, stages=3).to(device)

# 复制共享部分权重
v.backbone.load_state_dict(pi.backbone.state_dict())
v.inst_emb.load_state_dict(pi.inst_emb.state_dict())
v.query.load_state_dict(pi.query.state_dict())
pi.eval(); v.eval()

S = GRID * GRID
imgs = to_t(data["images"][:200], device)
inst = to_t(data["inst"][:200], device)

with torch.no_grad():
    pi_logits = pi(imgs, inst)                       # (200, S)
    cells = torch.arange(S, device=device).repeat(200, 1)   # (200, S)
    img_rep = imgs.repeat_interleave(S, dim=0)
    inst_rep = inst.repeat_interleave(S, dim=0)
    v_scores = v(img_rep, inst_rep, cells.reshape(-1)).reshape(200, S)

same = (pi_logits.argmax(-1) == v_scores.argmax(-1)).float().mean().item()
diff = (pi_logits - v_scores).abs()
print(f"argmax 一致率: {same:.4f}")
print(f"score 最大绝对差: {diff.max().item():.6f}")
print(f"score 平均绝对差: {diff.mean().item():.6f}")
print(f"pi_logits 前 3 行: {pi_logits[:1,:6].cpu().numpy()}")
print(f"v_scores  前 3 行: {v_scores[:1,:6].cpu().numpy()}")
