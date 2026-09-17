#!/usr/bin/env python3
"""
根据 eval_results.json 填充 EVAL_REPORT_TEMPLATE.md
生成最终报告：EVAL_REPORT.md
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path

TEMPLATE = Path("EVAL_REPORT_TEMPLATE.md")
RESULTS = Path("eval_results.json")
OUTPUT = Path("EVAL_REPORT.md")

def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"

def main():
    with open(RESULTS, encoding="utf-8") as f:
        r = json.load(f)

    with open(TEMPLATE, encoding="utf-8") as f:
        tmpl = f.read()

    # 基础信息
    tmpl = tmpl.replace("{{DATE}}", datetime.now().strftime("%Y-%m-%d %H:%M"))
    tmpl = tmpl.replace("{{AUTHOR}}", "auto-generated")
    tmpl = tmpl.replace("{{COMMIT_HASH}}", get_git_hash())
    tmpl = tmpl.replace("{{N_TEST}}", str(r["n_samples"]))
    tmpl = tmpl.replace("{{DEVICE}}", "CUDA (RTX 3060)")

    # 核心指标
    tmpl = tmpl.replace("{{ACTION_ACC:.4f}}", f"{r['action_accuracy']:.4f}")
    tmpl = tmpl.replace("{{ACTION_F1:.4f}}", f"{r['action_macro_f1']:.4f}")
    tmpl = tmpl.replace("{{COORD_MAE:.6f}}", f"{r['coord_mae']:.6f}")
    tmpl = tmpl.replace("{{SCROLL_ACC:.4f}}", f"{r['scroll_accuracy']:.4f}")

    # 状态判定
    def status(val, baseline, higher_better=True):
        if higher_better:
            return "✅ PASS" if val >= baseline - 1e-6 else "❌ FAIL"
        else:
            return "✅ PASS" if val <= baseline + 1e-6 else "❌ FAIL"

    tmpl = tmpl.replace("{{ACTION_STATUS}}", status(r["action_accuracy"], 1.0))
    tmpl = tmpl.replace("{{F1_STATUS}}", status(r["action_macro_f1"], 1.0))
    tmpl = tmpl.replace("{{COORD_STATUS}}", status(r["coord_mae"], 0.10, higher_better=False))
    tmpl = tmpl.replace("{{SCROLL_STATUS}}", status(r["scroll_accuracy"], 0.90))

    # 类别级
    cls = r["per_class_action"]
    for name in ["click", "type", "scroll", "move", "hotkey", "press"]:
        tmpl = tmpl.replace(f"{{{{{name.upper()}_N}}}}", str(cls[name]["count"]))
        tmpl = tmpl.replace(f"{{{{{name.upper()}_ACC:.4f}}}}", f"{cls[name]['accuracy']:.4f}")

    # 混淆矩阵
    cm = r["confusion_matrix"]
    for i in range(6):
        for j in range(6):
            tmpl = tmpl.replace(f"{{{{CM_{i}{j}}}}}", str(cm[i][j]))

    # 整体评估
    if r["action_accuracy"] >= 0.95 and r["coord_mae"] <= 0.10:
        overall = "优秀 —— 动作分类完美，坐标回归在可接受范围"
    elif r["action_accuracy"] >= 0.85:
        overall = "良好 —— 主要指标达标，坐标有提升空间"
    else:
        overall = "需改进 —— 核心指标未达标"
    tmpl = tmpl.replace("{{OVERALL_ASSESSMENT}}", overall)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(tmpl)

    print(f"Report generated: {OUTPUT}")

if __name__ == "__main__":
    main()