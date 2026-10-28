"""
real_site_oracle_audit：**量具审计** —— 把"状态折叠/视口外"拆成"真损失"与"oracle 错选"

起因（取证）：§3.7 报的 mdn "折叠头"（search y=-320）实为**无障碍隐藏菜单**（UL.a11y-menu 高 0）；
mdn logo 命中的是**页脚** logo；wikipedia home 命中的是正文里 13000px 处的 "Main page" 内容链接。
→ 所谓"状态折叠损失"混着 oracle 错选；像素轮的"视口外 25% 不可达"同样被污染。

本脚本对每个 (site, task)：
  1) 收集**所有**名字/角色匹配的候选（含已有的 v3 字段 + 逐候选的父链体检）；
  2) v1 = 现行 oracle_pick；v2 = 加"意图/可见性"过滤后的最优候选：
       · 不在隐藏容器里（祖先链上有 height==0 或 overflow:hidden/clip 且裁剪到不可见）；
       · 在文档范围内（0 ≤ y，且 y+h ≤ docHeight）；
       · home/logo 要求**头部区域**（y < 200 且不在 <footer> 内）；
       · search 要求 input/textarea/role=searchbox；
  3) 分类 v1 的错法（hidden-container / footer / deep-content / 无差），
     并给出 v2 下的 found / 状态类（in_view / 需滚动 / 仍不可达）。

用法: python experiments/web/real_site_oracle_audit.py
"""
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

from real_site_agent import SITES, TASKS, oracle_pick
from real_site_reach import snapshot_v3, flat_table, HIT_JS

CAND_JS = r"""
(rid) => {
  const el = document.querySelector('[data-reach-id="' + rid + '"]');
  if (!el) return null;
  const r = el.getBoundingClientRect();
  let hidden = false, inFooter = false, zeroAnc = null;
  let n = el;
  while (n && n !== document.documentElement) {
    const nr = n.getBoundingClientRect();
    const st = getComputedStyle(n);
    if (n.tagName === 'FOOTER') inFooter = true;
    const clipped = (st.overflow === 'hidden' || st.overflow === 'clip'
                     || st.clipPath !== 'none');
    const zero = nr.height < 2 || nr.width < 2;
    if (clipped && zero) { hidden = true; zeroAnc = n.tagName + '.' + String(n.className||'').slice(0,40); break; }
    n = n.parentElement;
  }
  return {x: r.x, y: r.y, w: r.width, h: r.height,
          hidden, zeroAnc, inFooter,
          doc: document.documentElement.scrollHeight,
          vh: innerHeight, vw: innerWidth,
          tag: el.tagName.toLowerCase(),
          role: el.getAttribute('role') || '',
          name: (el.getAttribute('aria-label') || el.getAttribute('placeholder')
                 || el.innerText || el.value || '').trim().slice(0, 50)};
}
"""


def classify(c, task):
    """v2 通过条件 + 状态类。"""
    if c is None:
        return "missing", None
    if c["hidden"]:
        return "hidden-container", None
    if c["y"] < -2 or c["y"] > c["doc"] or c["y"] + c["h"] < 0:
        return "out-of-doc", None
    if task in ("home", "logo") and (c["y"] > 200 or c["inFooter"]):
        return ("footer" if c["inFooter"] else "deep-content"), None
    if task == "search" and not (c["tag"] in ("input", "textarea")
                                 or c["role"] in ("searchbox", "combobox")):
        return "not-a-searchbox", None
    if c["y"] + c["h"] <= c["vh"] and c["y"] >= 0 and c["x"] >= 0 and c["x"] + c["w"] <= c["vw"]:
        return "ok", "in_view"
    return "ok", "needs_scroll"


def main():
    out = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        for name, url in SITES.items():
            pg = b.new_page(viewport={"width": 1000, "height": 700})
            try:
                pg.goto(url, wait_until="load", timeout=30000)
                pg.wait_for_timeout(1500)
            except Exception as e:
                print(f"[{name}] load FAIL {type(e).__name__}")
                pg.close()
                continue
            v3, _, _ = snapshot_v3(pg)
            tab = flat_table(v3)
            rec = dict(site=name, tasks=[])
            for task in TASKS:
                v1i = oracle_pick(tab, task)
                row = dict(task=task, v1=None, v1_class=None, v2=None, v2_class=None,
                           v2_state=None, v1_is_v2=None)
                if v1i >= 0:
                    c1 = pg.evaluate(CAND_JS, v3[v1i]["rid"])
                    row["v1"] = dict(rect=[round(c1["x"]), round(c1["y"]),
                                           round(c1["w"]), round(c1["h"])],
                                     tag=c1["tag"], name=c1["name"][:30])
                    cls, _ = classify(c1, task)
                    row["v1_class"] = cls
                # 在**同一 oracle 规则**的候选集里找第一个 v2 通过的
                # （做法：把表按规则过滤——复用 oracle_pick 的逻辑需要逐个试探，
                #   这里直接扫全表、用任务模式筛名字，再体检）
                pat = TASKS[task]["pattern"]
                cands = []
                for i, e in enumerate(v3):
                    blob = f"{e['name']} {e['role']}"
                    if task == "logo":
                        if not e.get("is_logo"):
                            continue
                    else:
                        if not pat.search(blob):
                            continue
                        if task == "search" and e["tag"] not in ("input", "textarea") \
                                and e["role"] not in ("searchbox", "combobox"):
                            continue
                    cands.append(i)
                for i in cands:
                    c = pg.evaluate(CAND_JS, v3[i]["rid"])
                    cls, state = classify(c, task)
                    if cls == "ok":
                        row["v2"] = dict(rect=[round(c["x"]), round(c["y"]),
                                               round(c["w"]), round(c["h"])],
                                         tag=c["tag"], name=c["name"][:30])
                        row["v2_class"] = "ok"
                        row["v2_state"] = state
                        row["v1_is_v2"] = bool(v1i == i)
                        break
                    if row["v2_class"] is None:
                        row["v2_class"] = cls        # 记第一个被拒的原因
                rec["tasks"].append(row)
            out.append(rec)
            n_ok = sum(1 for t in rec["tasks"] if t["v2"])
            print(f"[{name}] v2 oracle 可得 {n_ok}/4；" +
                  " ".join(f"{t['task']}:v1={t['v1_class'] or '-'}→v2={t['v2_state'] or '-'}"
                           for t in rec["tasks"]))
            pg.close()
        b.close()

    Path("results/web").mkdir(parents=True, exist_ok=True)
    Path("results/web/real_site_oracle_audit_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    # 汇总错法分布
    allt = [t for r in out for t in r["tasks"]]
    from collections import Counter
    c1 = Counter(t["v1_class"] for t in allt)
    changed = sum(1 for t in allt if t["v1"] and t["v2"] and not t["v1_is_v2"])
    print(f"\nv1 分类分布: {dict(c1)}")
    print(f"v1 与 v2 不同（改判）: {changed}/{sum(1 for t in allt if t['v1'] and t['v2'])}")
    print("已写出 results/web/real_site_oracle_audit_results.json")


if __name__ == "__main__":
    main()
