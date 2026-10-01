"""
WinSim W3：故障注入 —— UI 漂移 / 模态弹窗 / Loading

判据：
  - 无故障：naive 与 robust 都成功（对照）
  - 注入故障：naive（一次观测 + 忽略弹窗/加载）失败，robust（每次重观察 + 关弹窗 + 等待）成功
  - 故障确定性可复现；clearFault 可清除

用法:
    python experiments/winsim/winsim_w3.py
"""
import hashlib
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "results" / "winsim"
URL = "file:///" + str((HERE / "winsim" / "index.html").resolve()).replace("\\", "/")
VIEW = {"width": 900, "height": 600}

FAULTS = {
    "none": [],
    "drift": [{"kind": "drift", "dx": 14}],
    "dialog": [{"kind": "dialog", "text": "Save changes before closing?"}],
    "loading": [{"kind": "loading", "delay": 3000}],
    "all": [{"kind": "drift", "dx": 14},
            {"kind": "dialog", "text": "Are you sure?"},
            {"kind": "loading", "delay": 3000}],
}


def find(els, role, name=None):
    for e in els:
        if e["role"] == role and (name is None or e["name"] == name):
            return e
    return None


def click_el(pg, e):
    pg.evaluate(f"WinSim.click({e['x'] + e['w'] / 2},{e['y'] + e['h'] / 2})")


def naive(pg):
    """一次观测后用缓存坐标盲点；不关弹窗、不等加载。"""
    pg.evaluate("WinSim.openApp('explorer')")
    els = pg.evaluate("WinSim.observe({channel:'uia'}).uia")
    click_el(pg, find(els, "file", "notes.md"))
    click_el(pg, find(els, "button", "Cut"))
    click_el(pg, find(els, "dir", "Documents"))
    click_el(pg, find(els, "button", "Paste"))


def robust(pg):
    """每步重观察；遇弹窗先关；最后等待加载。"""
    pg.evaluate("WinSim.openApp('explorer')")

    def act(role, name=None):
        for _ in range(4):
            els = pg.evaluate("WinSim.observe({channel:'uia'}).uia")
            ok = find(els, "button", "OK")
            if ok:
                click_el(pg, ok)
                continue
            e = find(els, role, name)
            if e:
                click_el(pg, e)
                return
            pg.evaluate("WinSim.wait(600)")
    act("file", "notes.md"); act("button", "Cut")
    act("dir", "Documents"); act("button", "Paste")
    pg.evaluate("WinSim.wait(4000)")          # 等 loading 落地


def run(pw, faults, policy, seed=7):
    b = pw.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport=VIEW, device_scale_factor=1)
    pg.goto(URL)
    pg.evaluate(f"WinSim.reset({seed})")
    pg.evaluate("WinSim.setTask('move_notes')")
    for f in faults:
        pg.evaluate("(s)=>WinSim.injectFault(s)", f)
    (naive if policy == "naive" else robust)(pg)
    ok = pg.evaluate("WinSim.checkTask()")
    tr = pg.evaluate("WinSim.truth()")
    png = pg.screenshot()
    b.close()
    return ok, float(tr["cost"]), tr, png


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    rows = {}
    with sync_playwright() as pw:
        print(f"{'scenario':<10}{'naive':>7}{'robust':>8}{'cost_r':>9}")
        print("-" * 36)
        for name, fs in FAULTS.items():
            ok_n, _, _, _ = run(pw, fs, "naive")
            ok_r, c_r, tr_r, _ = run(pw, fs, "robust")
            rows[name] = dict(naive=ok_n, robust=ok_r, robust_cost=c_r)
            print(f"{name:<10}{str(ok_n):>7}{str(ok_r):>8}{c_r:>9.2f}")
            if name != "none":
                assert not ok_n and ok_r, f"{name}: 期望 naive 失败 robust 成功"

        # 确定性 + 清除
        _, _, _, p1 = run(pw, FAULTS["all"], "robust")
        _, _, _, p2 = run(pw, FAULTS["all"], "robust")
        det = sha(p1) == sha(p2)
        # clearFault 后 naive 应恢复
        b = pw.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page(viewport=VIEW)
        pg.goto(URL); pg.evaluate("WinSim.reset(7)"); pg.evaluate("WinSim.setTask('move_notes')")
        fid = pg.evaluate("WinSim.injectFault({kind:'dialog',text:'x'})")
        pg.evaluate(f"WinSim.clearFault({fid})")
        cleared = pg.evaluate("WinSim.truth().dialog") is None
        b.close()

    ok_all = all(rows[k]["naive"] and rows[k]["robust"] for k in ("none",)) and \
        all((not rows[k]["naive"]) and rows[k]["robust"]
            for k in ("drift", "dialog", "loading", "all")) and det and cleared
    print("-" * 36)
    print(f"确定性={det}  clearFault 生效={cleared}")
    print(f"\nW3 判据：无故障都成功；有故障 naive 失败 robust 成功；确定且可清除 -> {ok_all}")
    print(f"W3 {'通过' if ok_all else '未通过'}")
    (OUT / "winsim_w3_results.json").write_text(
        json.dumps({"scenarios": rows, "deterministic": det,
                    "clear_fault_works": cleared}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("已写出 winsim_w3_results.json")


if __name__ == "__main__":
    main()
