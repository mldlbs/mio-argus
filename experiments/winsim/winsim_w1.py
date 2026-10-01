"""
WinSim W1：真实应用状态 + 外部真值

- FileExplorer：真实文件树（导航/选择/Cut/Paste），成功由 fs 判定
- Settings    ：真实设置项（toggle），成功由 settings 判定
- 教师通道    ：WinSim.elements() 给可交互元素（含坐标）；oracle 用坐标点击完成
- 证伪门      ：同 seed 两次逐字节相同；且成功必须由 truth() 判定，不是"点了就算"

用法:
    python experiments/winsim/winsim_w1.py
"""
import hashlib
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "results" / "winsim"
URL = "file:///" + str((HERE / "winsim" / "index.html").resolve()).replace("\\", "/")
VIEW = {"width": 900, "height": 600}


def _els(pg):
    return pg.evaluate("WinSim.elements()")


def find(pg, role, name=None, app=None):
    for e in _els(pg):
        if e["role"] == role and (name is None or e["name"] == name) \
                and (app is None or e["app"] == app):
            return e
    return None


def click_el(pg, e):
    pg.evaluate(f"WinSim.click({e['x'] + e['w'] / 2},{e['y'] + e['h'] / 2})")


def oracle(pg, task):
    if task == "move_notes":
        pg.evaluate("WinSim.openApp('explorer')")
        click_el(pg, find(pg, "file", "notes.md"))
        click_el(pg, find(pg, "button", "Cut"))
        click_el(pg, find(pg, "dir", "Documents"))
        click_el(pg, find(pg, "button", "Paste"))
    elif task == "wifi_off":
        pg.evaluate("WinSim.openApp('settings')")
        click_el(pg, find(pg, "switch", "Wi-Fi"))
    pg.evaluate("WinSim.step(1000)")


def run(pw, task, seed, policy):
    b = pw.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport=VIEW, device_scale_factor=1)
    pg.goto(URL)
    pg.evaluate(f"WinSim.reset({seed})")
    pg.evaluate(f"WinSim.setTask('{task}')")
    if policy == "oracle":
        oracle(pg, task)
    elif policy == "noop":
        pg.evaluate("WinSim.step(1000)")
    ok = pg.evaluate("WinSim.checkTask()")
    tr = pg.evaluate("WinSim.truth()")
    png = pg.screenshot()
    b.close()
    return ok, tr, png


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    out = {}
    with sync_playwright() as pw:
        for task in ("move_notes", "wifi_off"):
            ok_o, tr_o, _ = run(pw, task, 7, "oracle")
            ok_n, _, _ = run(pw, task, 7, "noop")
            # 确定性：同一 seed+oracle 两次
            _, _, p1 = run(pw, task, 7, "oracle")
            _, _, p2 = run(pw, task, 7, "oracle")
            det = sha(p1) == sha(p2)
            key = "notes.md in Documents" if task == "move_notes" else "wifi"
            val = (("notes.md" in tr_o["fs"]["children"]["Documents"]["children"]
                    and "notes.md" not in tr_o["fs"]["children"])
                   if task == "move_notes" else tr_o["settings"]["wifi"])
            print(f"[{task}] oracle={ok_o} noop={ok_n} deterministic={det} "
                  f"truth({key})={val}")
            out[task] = dict(oracle=ok_o, noop=ok_n, deterministic=det, truth=val)

    allok = all(v["oracle"] and not v["noop"] and v["deterministic"]
                for v in out.values())
    print(f"\nW1 判据：oracle 全成功 且 noop 全失败 且 确定性成立 -> {allok}")
    print(f"W1 {'通过' if allok else '未通过'}")
    (OUT / "winsim_w1_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("已写出 winsim_w1_results.json")


if __name__ == "__main__":
    main()
