"""Bigemap 无特权执行基准：日志 schema 校验 + 汇总（不驱动 UI，不依赖 Bigemap）。

协议见 docs/BIGEMAP_EXEC_BENCH.md。runner 负责产出 JSONL（一行一步 + 一行一 trial），
本模块负责：字段校验、时间闭合断言、Wilson 区间、按臂/任务/控制类的汇总。

用法::

    python experiments/computer_use/exec_bench.py --self-test
    python experiments/computer_use/exec_bench.py --log results/computer_use/bigemap_bench_B_20261106.jsonl
    python experiments/computer_use/exec_bench.py --log a.jsonl --log b.jsonl --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("A", "B", "C", "S")   # S = 窗口层等两臂共用项
TASKS = ("T1", "T2", "T3", "T4", "T5", "T6")
LAYERS = ("L0", "L1", "L2", "L3", "L4")
CONTROL_CLASSES = ("window", "standard", "table_row", "canvas")
SEGMENTS = ("reset", "perceive", "decide", "act", "verify", "recover")
FAILURE_CLASSES = (
    "none", "focus_eaten", "stale_rect", "wrong_control", "no_pattern",
    "synthetic_rejected", "state_lost", "env", "timeout", "vacuous",
)
STEP_REQUIRED = (
    "kind", "trial", "arm", "task", "attempt", "step", "layer", "control_class",
    "t0", "t1", "window", "control", "action", "expect", "observed", "verdict",
    "silent", "failure_class", "segment_ms", "wall_ms",
)
TRIAL_REQUIRED = (
    "kind", "trial", "arm", "task", "success", "attempts", "takeovers",
    "reset_ok", "reset_ms", "wall_ms",
)
CLOSURE_TOL_MS = 50.0


def validate(rec: dict) -> list[str]:
    """返回错误列表；空列表 = 通过。"""
    errs: list[str] = []
    kind = rec.get("kind")
    if kind == "step":
        errs += [f"missing:{k}" for k in STEP_REQUIRED if k not in rec]
        if rec.get("arm") not in ARMS:
            errs.append(f"bad_arm:{rec.get('arm')}")
        if rec.get("task") not in TASKS:
            errs.append(f"bad_task:{rec.get('task')}")
        if rec.get("layer") not in LAYERS:
            errs.append(f"bad_layer:{rec.get('layer')}")
        if rec.get("control_class") not in CONTROL_CLASSES:
            errs.append(f"bad_control_class:{rec.get('control_class')}")
        if rec.get("verdict") not in ("pass", "fail", "vacuous", "unknown"):
            errs.append(f"bad_verdict:{rec.get('verdict')}")
        if rec.get("failure_class") not in FAILURE_CLASSES:
            errs.append(f"bad_failure_class:{rec.get('failure_class')}")
        if rec.get("verdict") == "fail" and rec.get("failure_class") == "none":
            errs.append("fail_without_failure_class")
        seg = rec.get("segment_ms") or {}
        if isinstance(seg, dict):
            unknown = set(seg) - set(SEGMENTS)
            if unknown:
                errs.append(f"unknown_segments:{sorted(unknown)}")
            try:
                total = sum(float(v) for v in seg.values())
                wall = float(rec.get("wall_ms") or 0.0)
                if abs(total - wall) > CLOSURE_TOL_MS:
                    errs.append(f"time_closure:{total:.0f}!={wall:.0f}")
            except (TypeError, ValueError):
                errs.append("segment_ms_not_numeric")
        else:
            errs.append("segment_ms_not_object")
        t0, t1 = rec.get("t0"), rec.get("t1")
        if isinstance(t0, (int, float)) and isinstance(t1, (int, float)):
            if t1 < t0:
                errs.append("t1_before_t0")
    elif kind == "trial":
        errs += [f"missing:{k}" for k in TRIAL_REQUIRED if k not in rec]
        if rec.get("arm") not in ARMS:
            errs.append(f"bad_arm:{rec.get('arm')}")
        if rec.get("task") not in TASKS:
            errs.append(f"bad_task:{rec.get('task')}")
        if rec.get("success") and not rec.get("reset_ok", True):
            errs.append("success_with_failed_reset")
    else:
        errs.append(f"bad_kind:{kind}")
    return errs


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 95% 区间。"""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - r) / d), min(1.0, (c + r) / d))


def _median(vals: list[float]) -> float:
    return round(statistics.median(vals), 1) if vals else float("nan")


def _iqr(vals: list[float]) -> tuple[float, float]:
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    q = statistics.quantiles(vals, n=4, method="inclusive")
    return (round(q[0], 1), round(q[2], 1))


def summarize(records: list[dict]) -> dict:
    steps = [r for r in records if r.get("kind") == "step"]
    trials = [r for r in records if r.get("kind") == "trial"]

    invalid = [r for r in trials if not r.get("reset_ok", True)]
    valid = [r for r in trials if r.get("reset_ok", True)]

    by_arm: dict[str, dict] = {}
    for arm in ARMS:
        rows = [r for r in valid if r.get("arm") == arm]
        if not rows:
            continue
        ok = sum(1 for r in rows if r.get("success"))
        lo, hi = wilson(ok, len(rows))
        per_task: dict[str, dict] = {}
        for task in TASKS:
            sub = [r for r in rows if r.get("task") == task]
            if not sub:
                continue
            k = sum(1 for r in sub if r.get("success"))
            tlo, thi = wilson(k, len(sub))
            succ = [r for r in sub if r.get("success")]
            retries = [max(0, r.get("attempts", 1) - 1) for r in succ]
            per_task[task] = {
                "n": len(sub), "success": k,
                "sr": round(k / len(sub), 3), "ci": [round(tlo, 3), round(thi, 3)],
                "retries_med": _median(retries), "retries_iqr": _iqr(retries),
                "wall_med": _median([float(r.get("wall_ms", 0)) / 1000 for r in succ]),
                "wall_iqr": _iqr([float(r.get("wall_ms", 0)) / 1000 for r in succ]),
                "takeovers": sum(int(r.get("takeovers", 0)) for r in sub),
            }
        by_arm[arm] = {
            "n": len(rows), "success": ok, "sr": round(ok / len(rows), 3),
            "ci": [round(lo, 3), round(hi, 3)],
            "invalid": sum(1 for r in invalid if r.get("arm") == arm),
            "tasks": per_task,
        }

    hr: dict[str, dict] = {}
    for arm in ARMS:
        for cls in CONTROL_CLASSES:
            sub = [r for r in steps if r.get("arm") == arm
                   and r.get("control_class") == cls
                   and r.get("layer") in ("L0", "L2")
                   and r.get("verdict") in ("pass", "fail")]
            if not sub:
                continue
            ok = sum(1 for r in sub if r.get("verdict") == "pass")
            lo, hi = wilson(ok, len(sub))
            hr.setdefault(arm, {})[cls] = {
                "n": len(sub), "hit": ok, "hr": round(ok / len(sub), 3),
                "ci": [round(lo, 3), round(hi, 3)],
            }

    judged = [r for r in steps if r.get("verdict") in ("pass", "fail")]
    silent = sum(1 for r in judged if r.get("silent"))
    failures = Counter(r.get("failure_class") for r in steps
                       if r.get("failure_class") not in (None, "none"))

    injected = [r for r in steps if (r.get("recovery") or {}).get("injected")]
    rr: dict[str, dict] = {}
    for key in sorted({f"{r['arm']}/{r['recovery']['injected']}" for r in injected}):
        sub = [r for r in injected if f"{r['arm']}/{r['recovery']['injected']}" == key]
        ok = sum(1 for r in sub if r["recovery"].get("ok"))
        rr[key] = {"n": len(sub), "ok": ok, "rr": round(ok / len(sub), 3)}

    steps_by_trial: dict[str, list[dict]] = defaultdict(list)
    for r in steps:
        steps_by_trial[r.get("trial")].append(r)

    seg_by: dict[str, dict] = {}
    for arm in ARMS:
        for task in TASKS:
            tids = [r["trial"] for r in valid
                    if r.get("arm") == arm and r.get("task") == task]
            if not tids:
                continue
            per_trial = []
            for tid in tids:
                sub = steps_by_trial.get(tid, [])
                per_trial.append({k: sum(float((r.get("segment_ms") or {}).get(k, 0.0))
                                         for r in sub) for k in SEGMENTS})
            seg_by[f"{arm}/{task}"] = {
                k: round(statistics.mean(d[k] for d in per_trial), 1) for k in SEGMENTS
            }

    return {
        "trials": {"total": len(trials), "valid": len(valid), "invalid": len(invalid),
                   "invalid_rate": round(len(invalid) / len(trials), 3) if trials else 0.0},
        "arms": by_arm,
        "hit_rate_by_control_class": hr,
        "silent_failure": {"judged_steps": len(judged), "silent": silent,
                           "sfr": round(silent / len(judged), 3) if judged else 0.0},
        "failure_classes": dict(failures.most_common()),
        "recovery": rr,
        "segment_mean_ms_by_task": seg_by,
    }


def render(summary: dict) -> str:
    out = ["# Bigemap 执行基准汇总", ""]
    t = summary["trials"]
    out += [f"试验：{t['total']}（有效 {t['valid']} / 无效 {t['invalid']}，"
            f"invalid 率 {t['invalid_rate']:.1%}）", ""]

    out += ["## 1. 任务成功率", "",
            "| 臂 | 任务 | n | 成功 | SR | 95% CI | 中位重试 [IQR] | 中位墙钟 s | 人工接管 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for arm, a in summary["arms"].items():
        for task, r in a["tasks"].items():
            out.append(f"| {arm} | {task} | {r['n']} | {r['success']} | {r['sr']:.1%} | "
                       f"[{r['ci'][0]:.1%}, {r['ci'][1]:.1%}] | "
                       f"{r['retries_med']} [{r['retries_iqr'][0]}, {r['retries_iqr'][1]}] | "
                       f"{r['wall_med']} | {r['takeovers']} |")
        out.append(f"| **{arm}** | 合计 | {a['n']} | {a['success']} | {a['sr']:.1%} | "
                   f"[{a['ci'][0]:.1%}, {a['ci'][1]:.1%}] | — | — | — |")
    out.append("")

    out += ["## 2. 控制类命中率（L2，不含 vacuous）", "",
            "| 臂 | 控制类 | n | 命中 | HR | 95% CI |", "|---|---|---|---|---|---|"]
    for arm, classes in summary["hit_rate_by_control_class"].items():
        for cls, r in classes.items():
            out.append(f"| {arm} | {cls} | {r['n']} | {r['hit']} | {r['hr']:.1%} | "
                       f"[{r['ci'][0]:.1%}, {r['ci'][1]:.1%}] |")
    out.append("")

    s = summary["silent_failure"]
    out += ["## 3. 静默失败", "",
            f"有判定步数 {s['judged_steps']}，静默失败 {s['silent']}，**SFR = {s['sfr']:.1%}**", ""]

    out += ["## 4. 失败归因", "", "| failure_class | 次数 |", "|---|---|"]
    for k, v in summary["failure_classes"].items():
        out.append(f"| {k} | {v} |")
    out.append("")

    out += ["## 5. 故障恢复", "", "| 故障 | 注入 | 恢复 | RR |", "|---|---|---|---|"]
    for f, r in summary["recovery"].items():
        out.append(f"| {f} | {r['n']} | {r['ok']} | {r['rr']:.1%} |")
    out.append("")

    out += ["## 6. 时间分段（均值 ms）", "",
            "| 臂/任务 | reset | perceive | decide | act | verify | recover |",
            "|---|---|---|---|---|---|---|"]
    for task, seg in summary["segment_mean_ms_by_task"].items():
        out.append("| " + task + " | " + " | ".join(f"{seg.get(k, 0.0)}" for k in SEGMENTS) + " |")
    out.append("")
    return "\n".join(out)


def load_jsonl(path: Path) -> list[dict]:
    recs = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i} JSON 解析失败：{e}")
    return recs


# ---------------------------------------------------------------- self-test

def _step(trial, arm, task, step, cls, verdict="pass", fc="none", silent=False,
          wall=380.0, injected=None, ok=True, attempt=1) -> dict:
    wall = float(wall)
    seg = {"perceive": round(wall * 0.30, 1), "decide": 0.0,
           "act": round(wall * 0.45, 1), "verify": 0.0}
    seg["verify"] = round(wall - sum(seg.values()), 1)
    rec = {
        "kind": "step", "trial": trial, "arm": arm, "task": task, "attempt": attempt,
        "step": step, "layer": "L2", "control_class": cls, "t0": 1.0, "t1": 1.0 + wall / 1000,
        "window": {"hwnd": 1, "rect": [1344, 16, 2544, 916], "minimized": False, "foreground": True},
        "control": {"name": "x", "type": "Button", "rect": [1, 1, 10, 10], "patterns": ["Invoke"]},
        "action": {"kind": "invoke", "source": "uia", "coords": None},
        "expect": {"pred": "dialog_open", "pre": "closed"},
        "observed": {"open": verdict == "pass"},
        "verdict": verdict, "silent": silent, "failure_class": fc,
        "recovery": {"injected": injected, "ok": ok} if injected else None,
        "segment_ms": seg, "wall_ms": wall,
    }
    return rec


def _self_test(seed: int = 7) -> dict:
    rng = random.Random(seed)
    recs = []

    # T5：6 步（4 standard + 2 table_row），trial 成功 = 全部步通过
    pass_p = {
        ("A", "standard"): 0.97, ("A", "table_row"): 0.85,
        ("B", "standard"): 1.00, ("B", "table_row"): 0.97,
    }
    for arm in ("A", "B"):
        for i in range(30):
            tid = f"T5-{arm}-{i:04d}"
            reset_ok = rng.random() > 0.03
            classes = ["standard", "standard", "table_row",
                       "standard", "table_row", "standard"]
            fails = retries = 0
            for s, cls in enumerate(classes):
                ok = rng.random() < pass_p[(arm, cls)]
                fc = "none" if ok else ("no_pattern" if cls == "table_row" else "wrong_control")
                if ok:
                    silent = rng.random() < (0.01 if arm == "A" else 0.002)
                    recs.append(_step(tid, arm, "T5", s, cls, silent=silent,
                                      wall=rng.randint(180, 420)))
                    continue
                recs.append(_step(tid, arm, "T5", s, cls, verdict="fail", fc=fc,
                                  wall=rng.randint(180, 420)))
                # 失败被环内 verifier 抓到并重试成功：trial 仍成功，但计入 retries
                if rng.random() < (0.50 if arm == "A" else 0.85):
                    retries += 1
                    recs.append(_step(tid, arm, "T5", s, cls, wall=rng.randint(180, 420),
                                      attempt=2))
                else:
                    fails += 1
            recs.append({
                "kind": "trial", "trial": tid, "arm": arm, "task": "T5",
                "success": fails == 0 and reset_ok, "attempts": 1 + retries + fails,
                "takeovers": 0, "reset_ok": reset_ok,
                "reset_ms": rng.randint(900, 1800),
                "wall_ms": rng.randint(55_000, 95_000) if arm == "A"
                           else rng.randint(45_000, 80_000),
            })

    # T4：画布双击，两臂都只能用鼠标 → 命中率应接近（共同下限）
    for arm in ("A", "B"):
        for i in range(20):
            tid = f"T4-{arm}-{i:04d}"
            reset_ok = rng.random() > 0.03
            ok = rng.random() < 0.62
            recs.append(_step(tid, arm, "T4", 0, "canvas",
                              verdict="pass" if ok else "fail",
                              fc="none" if ok else "synthetic_rejected",
                              wall=rng.randint(400, 1200)))
            recs.append({
                "kind": "trial", "trial": tid, "arm": arm, "task": "T4",
                "success": ok and reset_ok, "attempts": 1 if ok else rng.randint(2, 4),
                "takeovers": 0, "reset_ok": reset_ok,
                "reset_ms": rng.randint(700, 1400),
                "wall_ms": rng.randint(6_000, 14_000),
            })

    # 故障注入（B 臂）：F1 最小化 / F6 stale rect
    for f, rate in (("F1", 0.90), ("F6", 0.75)):
        for i in range(10):
            ok = rng.random() < rate
            recs.append(_step(f"R-{f}-{i}", "B", "T3", 1, "table_row",
                              verdict="pass" if ok else "fail",
                              fc="none" if ok else ("state_lost" if f == "F1" else "stale_rect"),
                              injected=f, ok=ok))

    summary = summarize(recs)

    # 自检断言：汇总自洽 + 脏记录会被抓出来
    assert summary["arms"]["B"]["sr"] > summary["arms"]["A"]["sr"]
    assert summary["hit_rate_by_control_class"]["B"]["table_row"]["hr"] >= \
        summary["hit_rate_by_control_class"]["A"]["table_row"]["hr"]
    assert 0.3 < summary["hit_rate_by_control_class"]["A"]["canvas"]["hr"] < 0.95
    assert summary["recovery"]["B/F1"]["rr"] > 0
    assert 0.0 <= summary["silent_failure"]["sfr"] <= 1.0

    bad = dict(recs[0])
    bad["segment_ms"] = {"act": 1.0}
    assert any(e.startswith("time_closure") for e in validate(bad)), "闭合断言失效"
    bad2 = dict(recs[0]); bad2.pop("control")
    assert any(e.startswith("missing:control") for e in validate(bad2))
    assert all(not validate(r) for r in recs), "self-test 样本本身不合法"
    return summary


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Bigemap 执行基准：日志校验与汇总")
    ap.add_argument("--log", action="append", default=[], help="JSONL 日志，可多次指定")
    ap.add_argument("--json", help="把汇总写到该路径")
    ap.add_argument("--self-test", action="store_true", help="用合成数据跑通汇总管线")
    args = ap.parse_args()

    if args.self_test:
        summary = _self_test()
        print("> 合成数据自检，仅验证 schema / 汇总 / 闭合断言，**不是实验结果**。\n")
        print(render(summary))
        print("[self-test] ok")
        return
    if not args.log:
        ap.error("需要 --log 或 --self-test")

    recs: list[dict] = []
    for p in args.log:
        recs += load_jsonl(Path(p))

    bad = [(i, validate(r)) for i, r in enumerate(recs) if validate(r)]
    if bad:
        print(f"[warn] {len(bad)} 条记录未通过校验，示例：", file=sys.stderr)
        for i, errs in bad[:3]:
            print(f"  #{i}: {errs}", file=sys.stderr)

    summary = summarize(recs)
    print(render(summary))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {args.json}")


if __name__ == "__main__":
    main()
