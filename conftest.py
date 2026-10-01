"""pytest 路径引导：让测试可以 import 仓库根、mio/、infra/ 下的模块。

阶段 2 目录迁移后，测试位于 experiments/computer_use/ 等子目录，
其同级模块由 pytest 自动加入 sys.path，但跨目录模块（mio_argus、
api_server 等）需要在此显式登记。
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "mio", _ROOT / "infra"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
