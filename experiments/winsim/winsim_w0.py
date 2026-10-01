"""
WinSim W0 证伪门：确定性

同一 seed 的脚本化操作序列，两次独立加载必须产出**逐字节相同**的截图。
不同 seed 必须不同（对照，证明不是恒等画面）。

用法:
    python experiments/winsim/winsim_w0.py
"""
import hashlib
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "results" / "winsim"
URL = "file:///" + str((HERE / "winsim" / "index.html").resolve()).replace("\\", "/")
VIEW = {"width": 900, "height": 600}

# 固定脚本（无时间依赖、无随机）
SCRIPT = [
    "WinSim.reset(SEED)",
    "WinSim.openApp('explorer')",
    "WinSim.openApp('settings')",
    "WinSim.moveWindow(2, 150, 90)",
    "WinSim.step(3*3600*1000)",        # 时钟 +3h
    "WinSim.openApp('notepad')",
    "WinSim.focusWindow(1)",
    "WinSim.maximizeWindow(3)",
    "WinSim.step(90*60*1000)",
]


def run(pw, seed):
    b = pw.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport=VIEW, device_scale_factor=1)
    pg.goto(URL)
    for op in SCRIPT:
        pg.evaluate(op.replace("SEED", str(seed)))
    png = pg.screenshot()
    obs = pg.evaluate("WinSim.observe()")
    b.close()
    return png, obs


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    with sync_playwright() as pw:
        a_png, a_obs = run(pw, 42)
        b_png, b_obs = run(pw, 42)
        c_png, _ = run(pw, 43)

    same = sha(a_png) == sha(b_png)
    obs_same = (a_obs == b_obs)
    diff_seed = sha(a_png) != sha(c_png)

    print(f"seed=42 run#1 sha={sha(a_png)[:16]} ({len(a_png)} B)")
    print(f"seed=42 run#2 sha={sha(b_png)[:16]} ({len(b_png)} B)")
    print(f"seed=43       sha={sha(c_png)[:16]} ({len(c_png)} B)")
    print(f"\n[门] 同 seed 两次逐字节相同 : {same}")
    print(f"[门] observe() 状态一致      : {obs_same}")
    print(f"[对照] 不同 seed 画面不同    : {diff_seed}")
    print(f"\nW0 {'通过' if (same and obs_same and diff_seed) else '未通过'}")

    (OUT / "winsim_w0_results.json").write_text(json.dumps(
        {"same_seed_bitwise": same, "observe_identical": obs_same,
         "different_seed_differs": diff_seed,
         "sha": {"42a": sha(a_png), "42b": sha(b_png), "43": sha(c_png)}},
        indent=2), encoding="utf-8")
    print("已写出 winsim_w0_results.json")


if __name__ == "__main__":
    main()
