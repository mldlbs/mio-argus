"""
WinSim W4：held-out 协议（故障级）—— 覆盖不足的量化

把 site_holdout 的思路搬进 WinSim，换一个 OOD 轴：**故障类型**。
  in-domain 故障：none, drift        （训练/调优时见过的）
  OOD  故障     ：dialog, loading    （没见过的）

三种策略：
  naive    （一次观测，什么都不处理）        —— 最弱
  inductive（每步重观察【只学会了 drift】，不关弹窗/不等加载） —— "过拟合 in-domain"
  robust   （重观察 + 关弹窗 + 等加载）      —— 通用
oracle 任务：move_notes（会同时触发 drift/dialog/loading）

判据：inductive 在 in-domain 故障上成功、在 OOD 故障上掉到 0；robust 两者皆成功。
       → 在 in-domain 上"做对了"，不代表能迁移。

用法:
    python experiments/winsim/winsim_w4.py
"""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "results" / "winsim"
URL = "file:///" + str((HERE / "winsim" / "index.html").resolve()).replace("\\", "/")
VIEW = {"width": 900, "height": 600}

FAULTS = {
    "none":    [],
    "drift":   [{"kind": "drift", "dx": 14}],
    "dialog":  [{"kind": "dialog", "text": "Save changes?"}],
    "loading": [{"kind": "loading", "delay": 3000}],
}
IN_DOMAIN = ("none", "drift")
OOD = ("dialog", "loading")


def find(els, role, name=None):
    for e in els:
        if e["role"] == role and (name is None or e["name"] == name):
            return e
    return None


def click_el(pg, e):
    pg.evaluate(f"WinSim.click({e['x'] + e['w'] / 2},{e['y'] + e['h'] / 2})")


def naive(pg):
    pg.evaluate("WinSim.openApp('explorer')")
    els = pg.evaluate("WinSim.observe({channel:'uia'}).uia")
    for r, n in [("file", "notes.md"), ("button", "Cut"),
                 ("dir", "Documents"), ("button", "Paste")]:
        click_el(pg, find(els, r, n))


def inductive(pg):
    """每步重观察（能扛 drift），但不处理 dialog/loading。"""
    pg.evaluate("WinSim.openApp('explorer')")

    def act(role, name=None):
        for _ in range(3):
            els = pg.evaluate("WinSim.observe({channel:'uia'}).uia")
            e = find(els, role, name)
            if e:
                click_el(pg, e)
                return
            pg.evaluate("WinSim.wait(300)")
    act("file", "notes.md"); act("button", "Cut")
    act("dir", "Documents"); act("button", "Paste")


def robust(pg):
    pg.evaluate("WinSim.openApp('explorer')")

    def act(role, name=None):
        for _ in range(4):
            els = pg.evaluate("WinSim.observe({channel:'uia'}).uia")
            ok = find(els, "button", "OK")
            if ok:
                click_el(pg, ok); continue
            e = find(els, role, name)
            if e:
                click_el(pg, e); return
            pg.evaluate("WinSim.wait(600)")
    act("file", "notes.md"); act("button", "Cut")
    act("dir", "Documents"); act("button", "Paste")
    pg.evaluate("WinSim.wait(4000)")


AGENTS = {"naive": naive, "inductive": inductive, "robust": robust}


def run(pw, fault, agent, seed=7):
    b = pw.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport=VIEW, device_scale_factor=1)
    pg.goto(URL)
    pg.evaluate(f"WinSim.reset({seed})")
    pg.evaluate("WinSim.setTask('move_notes')")
    for f in FAULTS[fault]:
        pg.evaluate("(s)=>WinSim.injectFault(s)", f)
    AGENTS[agent](pg)
    ok = pg.evaluate("WinSim.checkTask()")
    b.close()
    return ok


def main():
    rows = {}
    with sync_playwright() as pw:
        print(f"{'fault':<9}{'split':<11}" +
              "".join(f"{a:>11}" for a in AGENTS))
        print("-" * 52)
        for fault in FAULTS:
            split = "in-domain" if fault in IN_DOMAIN else "OOD"
            res = {}
            for a in AGENTS:
                res[a] = run(pw, fault, a)
            rows[fault] = res
            print(f"{fault:<9}{split:<11}" +
                  "".join(f"{str(res[a]):>11}" for a in AGENTS))

        print("-" * 52)
        def split_acc(agent, split):
            ks = IN_DOMAIN if split == "in-domain" else OOD
            return sum(rows[k][agent] for k in ks) / len(ks)
        summ = {a: dict(in_domain=split_acc(a, "in-domain"),
                        ood=split_acc(a, "OOD")) for a in AGENTS}
        for a in AGENTS:
            s = summ[a]
            print(f"  {a:<10} in-domain={s['in_domain']:.2f} "
                  f"OOD={s['ood']:.2f} drop={s['in_domain'] - s['ood']:+.2f}")
        ok_protocol = (summ["inductive"]["in_domain"] == 1.0
                       and summ["inductive"]["ood"] == 0.0
                       and summ["robust"]["in_domain"] == 1.0
                       and summ["robust"]["ood"] == 1.0)
        print(f"\nW4 判据：inductive 在 OOD 上掉到 0，robust 两者皆 1 -> {ok_protocol}")
        print(f"W4 {'通过' if ok_protocol else '未通过'}")

    (OUT / "winsim_w4_results.json").write_text(
        json.dumps({"matrix": rows, "summary": summ,
                    "in_domain_faults": list(IN_DOMAIN),
                    "ood_faults": list(OOD)}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("已写出 winsim_w4_results.json")


if __name__ == "__main__":
    main()
