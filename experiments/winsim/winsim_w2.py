"""
WinSim W2：双通道观测 + 成本模型

- observe({channel:'uia'})   特权结构通道（便宜，含坐标）
- observe({channel:'pixels'}) 像素通道（贵；真实部署只有这个）
- 成本：观察/动作/等待都计费（元认知需要）
- 证伪门：盲操作（不观察）应失败；且成本可确定性复现

用法:
    python experiments/winsim/winsim_w2.py
"""
import hashlib
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "results" / "winsim"
URL = "file:///" + str((HERE / "winsim" / "index.html").resolve()).replace("\\", "/")
VIEW = {"width": 900, "height": 600}


def crown(pg, channel):
    """返回该通道下的可交互元素。uia=特权直接给；pixels=需视觉（此处用完美视觉替身）。"""
    if channel == "uia":
        return pg.evaluate("WinSim.observe({channel:'uia'}).uia")
    pg.evaluate("WinSim.observe({channel:'pixels'})")     # 计像素观测费
    return pg.evaluate("WinSim.elements()")               # 假定的完美像素接地


def find(els, role, name=None):
    for e in els:
        if e["role"] == role and (name is None or e["name"] == name):
            return e
    return None


def click_el(pg, e):
    pg.evaluate(f"WinSim.click({e['x'] + e['w'] / 2},{e['y'] + e['h'] / 2})")


def act(pg, channel, role, name=None):
    click_el(pg, find(crown(pg, channel), role, name))


def oracle(pg, task, channel):
    if task == "move_notes":
        pg.evaluate("WinSim.openApp('explorer')")
        act(pg, channel, "file", "notes.md")
        act(pg, channel, "button", "Cut")
        act(pg, channel, "dir", "Documents")
        act(pg, channel, "button", "Paste")
    elif task == "wifi_off":
        pg.evaluate("WinSim.openApp('settings')")
        act(pg, channel, "switch", "Wi-Fi")
    pg.evaluate("WinSim.wait(500)")


def blind(pg, task):
    """不观察，按硬编码坐标盲点（窗口位置随 seed 变 → 应失败）。"""
    pg.evaluate("WinSim.openApp('explorer')")
    for (x, y) in [(60, 40), (100, 120), (100, 90), (100, 120), (140, 60)]:
        pg.evaluate(f"WinSim.click({x},{y})")
    pg.evaluate("WinSim.wait(500)")


def run(pw, task, seed, policy, channel=None):
    b = pw.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport=VIEW, device_scale_factor=1)
    pg.goto(URL)
    pg.evaluate(f"WinSim.reset({seed})")
    pg.evaluate(f"WinSim.setTask('{task}')")
    if policy == "oracle":
        oracle(pg, task, channel)
    else:
        blind(pg, task)
    ok = pg.evaluate("WinSim.checkTask()")
    cost = pg.evaluate("WinSim.truth().cost")
    png = pg.screenshot()
    b.close()
    return ok, float(cost), png


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    out = {}
    with sync_playwright() as pw:
        print(f"{'task':<12}{'policy':<22}{'ok':>5}{'cost':>8}")
        print("-" * 48)
        for task in ("move_notes", "wifi_off"):
            ok_u, c_u, _ = run(pw, task, 7, "oracle", "uia")
            ok_p, c_p, _ = run(pw, task, 7, "oracle", "pixels")
            ok_b, c_b, _ = run(pw, task, 7, "blind")
            # 成本确定性
            _, c_u2, p1 = run(pw, task, 7, "oracle", "uia")
            _, _, p2 = run(pw, task, 7, "oracle", "uia")
            det = (c_u == c_u2) and (sha(p1) == sha(p2))
            print(f"{task:<12}{'oracle/uia':<22}{str(ok_u):>5}{c_u:>8.2f}")
            print(f"{task:<12}{'oracle/pixels':<22}{str(ok_p):>5}{c_p:>8.2f}")
            print(f"{task:<12}{'blind(no observe)':<22}{str(ok_b):>5}{c_b:>8.2f}")
            out[task] = dict(uia=dict(ok=ok_u, cost=c_u),
                             pixels=dict(ok=ok_p, cost=c_p),
                             blind=dict(ok=ok_b, cost=c_b),
                             cost_deterministic=det)

        print("-" * 48)
        gap = {t: out[t]["pixels"]["cost"] - out[t]["uia"]["cost"] for t in out}
        print(f"像素 vs 结构 成本差：{ {k: round(v,2) for k,v in gap.items()} }")
        allok = all(o["uia"]["ok"] and o["pixels"]["ok"] and not o["blind"]["ok"]
                    and o["cost_deterministic"] for o in out.values())
        print(f"\nW2 判据：两通道都成功 且 盲操作失败 且 成本可复现 -> {allok}")
        print(f"W2 {'通过' if allok else '未通过'}")

    (OUT / "winsim_w2_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 winsim_w2_results.json")


if __name__ == "__main__":
    main()
