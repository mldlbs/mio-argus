"""
QGIS W22：同一端到端链上的对照 —— "修低层" vs "加验证" vs "盲跑"，以及语言层的错误 V 抓不到

W18→W19 的故事：选择段的 Run 点击有 ~25% 概率被"激活窗口"吃掉（静默失败）。
三种处理方式：
  · blind  ：点一下就往下走（不验证）           → 期望 ~75%
  · v-gated：真鼠标点击 + 验证选择数 + 重试      → 期望 100%，但每次重试 +6 s
  · invoke ：低层修复（UIA Invoke，不需要焦点）  → 期望 100%，且无重试代价
同一条链：指令 → 表达式 → Processing 选择 → 右键导出 → 文件真值（要素集合）。

另测**语言层错误**：learned 对未见句式产出错表达式
  · 无效表达式（`"a" > 3000`）→ 算法报错 → 选择数不变 → 步骤级 V **能**抓到；
  · 合法但语义错的表达式（把 "不是 West" 译成 `"name" = 'West'`）→ 选择"成功"（1 个），
    步骤级 V（选择数符合该表达式的预期）**看不出来**，只有**按指令语义**核验才能发现。

用法: python experiments/qgis/qgis_bridge/run_w22.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from client import call  # noqa: E402
from run_w16 import export_via_gui, features_of  # noqa: E402
from run_w18 import select_via_gui, SCENE  # noqa: E402

ARD = "area" if False else None
OUT = HERE / "data" / "w22.geojson"
TRIALS = 8
EXPR_OK = '"area" > 400'
WANT_OK = {"North"}
ARMS = [("blind", "blind"), ("v-gated", "verify"), ("invoke", "invoke")]

# 语言层：model 产出的错表达式（W21 实测）
LANG_CASES = [
    dict(instr="凡是面积至少300的区都导出来", expr='"a" > 3000',
         want_by_instr={"North", "South", "Central"}, note="无效表达式（字段 a 不存在）"),
    dict(instr="名称不是West的区请导出", expr='"name" = \'West\'',
         want_by_instr={"North", "South", "Central", "East"}, note="合法但语义反了"),
]


def expr_expected_features(expr):
    """按表达式求它在 5 个区域上的预期集合（步骤级 V 的"预期"）。"""
    FEATS = {"North": 500, "South": 400, "Central": 300, "West": 150, "East": 120}
    import re
    m = re.match(r'"(\w+)"\s*(>=|<=|!=|=|>|<)\s*([\w\']+)', expr)
    if not m:
        return None
    f, op, v = m.group(1), m.group(2), m.group(3).strip("'")
    def keep(n):
        a = FEATS[n]
        try:
            vv = int(v)
            return {">": a > vv, "<": a < vv, ">=": a >= vv, "<=": a <= vv}.get(op, False)
        except ValueError:
            return {"=": n == v, "!=": n != v}.get(op, False)
    return {n for n in FEATS if keep(n)}


def main():
    call({"op": "win", "maximize": True})
    print(f"端到端链（{TRIALS} 次/臂）：指令 → 表达式 → Processing 选择 → GUI 导出 → 文件真值\n")
    print(f"{'arm':>8}{'成功/总':>9}{'选择ok':>7}{'兼容时间ms':>11}{'选择ms':>9}{'导出ms':>9}")
    out = {}
    for arm, mode in ARMS:
        oks = 0
        sels = 0
        ts, sel_ts, exp_ts = [], [], []
        for i in range(TRIALS):
            call({"op": "reset", "scene": SCENE})
            time.sleep(0.3)
            t0 = time.time()
            ok_sel, st_sel = select_via_gui(EXPR_OK, expect_n=len(WANT_OK), run_mode=mode)
            if OUT.exists():
                OUT.unlink()
            ok_exp, st_exp, note = export_via_gui(OUT, selected_only=True)
            got = features_of(OUT)
            good = bool(ok_exp and got == WANT_OK)
            oks += int(good); sels += int(ok_sel)
            ts.append((time.time() - t0) * 1000)
            sel_ts.append(st_sel.get("total_ms", 0) if st_sel else 0)
            exp_ts.append(st_exp.get("total_ms", 0))
            print(f"    [{arm}] {i+1}: 选择ok={ok_sel} 文件要素={sorted(got)} 成功={good}")
        out[arm] = dict(success=f"{oks}/{TRIALS}", sel_ok=f"{sels}/{TRIALS}",
                        ms_median=float(np.median(ts)),
                        sel_ms_median=float(np.median(sel_ts)),
                        exp_ms_median=float(np.median(exp_ts)))
        print(f"{arm:>8}{oks:>7}/{TRIALS}{sels:>7}{np.median(ts):>11.0f}"
              f"{np.median(sel_ts):>9.0f}{np.median(exp_ts):>9.0f}")

    print("\n语言层错误能否被步骤级 V 抓到：")
    lang_rows = []
    for c in LANG_CASES:
        call({"op": "reset", "scene": SCENE})
        time.sleep(0.3)
        exp_sel = expr_expected_features(c["expr"])
        ok_sel, st_sel = select_via_gui(c["expr"], expect_n=len(exp_sel or []), run_mode="invoke")
        if OUT.exists():
            OUT.unlink()
        ok_exp, st_exp, note = export_via_gui(OUT, selected_only=True)
        got = features_of(OUT)
        step_v_pass = bool(ok_sel and got == exp_sel)          # 步骤级 V：选择是否符合"该表达式"
        instr_pass = bool(got == c["want_by_instr"])            # 指令级真值
        lang_rows.append(dict(instr=c["instr"], expr=c["expr"], note=c["note"],
                              expr_sel=sorted(exp_sel) if exp_sel else None,
                              file=sorted(got), want_by_instr=sorted(c["want_by_instr"]),
                              step_v=step_v_pass, instr_truth=instr_pass))
        print(f"  {c['instr']!r}\n     expr={c['expr']!r}（{c['note']}）\n"
              f"     步骤级V={'通过' if step_v_pass else '拦下'}  指令级真值={'对' if instr_pass else '错'}"
              f"  文件={sorted(got)}")
    n_caught = sum(1 for r in lang_rows if not r["step_v"])
    print(f"  → 步骤级 V 拦下 {n_caught}/{len(lang_rows)}（另一条只有指令级核验才能发现）")

    (HERE / "w22_results.json").write_text(json.dumps(
        {"arms": out, "language": lang_rows, "trials": TRIALS}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("已写出 experiments/qgis/qgis_bridge/w22_results.json")


if __name__ == "__main__":
    main()
