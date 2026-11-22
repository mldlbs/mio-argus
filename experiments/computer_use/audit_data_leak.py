"""数据泄露审计（A0 前置）：指令里是否直接写出了目标坐标？

若一条样本的 instruction 里同时出现 (x, y) 的近似数值，模型可以只看文本答对，
视觉通路拿不到梯度——本项目真实的翻车原因（168/168）。

用法：
    python experiments/computer_use/audit_data_leak.py            # 扫 artifacts/ 下所有 training_data*.json
    python experiments/computer_use/audit_data_leak.py --path X   # 指定文件
判定：
    泄露率 > 30%  → 该数据集不得用于定位训练（先清洗指令模板）
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NUM = re.compile(r"\b(\d{2,4})\b")
TOL = 3   # px


def scan(path: Path):
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                                    # noqa: BLE001
        return {"path": str(path), "error": str(exc)}
    rows = d if isinstance(d, list) else d.get("samples", d.get("data", []))
    if not isinstance(rows, list):
        return {"path": str(path), "error": "无法识别的结构"}
    with_xy = [r for r in rows if isinstance(r, dict) and isinstance(r.get("params"), dict)
               and "x" in r["params"]]
    leak = 0
    for r in with_xy:
        nums = [int(n) for n in NUM.findall(str(r.get("instruction", "")))]
        x, y = r["params"].get("x", 0), r["params"].get("y", 0)
        if any(abs(n - x) <= TOL for n in nums) and any(abs(n - y) <= TOL for n in nums):
            leak += 1
    rate = leak / max(len(with_xy), 1)
    return {"path": str(path), "rows": len(rows), "with_xy": len(with_xy),
            "leak": leak, "leak_rate": round(rate, 4),
            "verdict": "LEAKED" if len(with_xy) and rate > 0.30 else "OK",
            "example": str(with_xy[0].get("instruction"))[:80] if with_xy else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="")
    ap.add_argument("--glob", default="artifacts/**/*training_data*.json")
    ap.add_argument("--out", default="results/computer_use/data_leak_audit.json")
    args = ap.parse_args()

    paths = ([Path(args.path)] if args.path
             else [Path(p) for p in sorted(glob.glob(str(ROOT / args.glob), recursive=True))])
    results = [r for r in (scan(p) for p in paths) if r.get("with_xy", 0) > 0 or "error" in r]

    print(f"{'dataset':<58} {'xy':>5} {'leak':>5} {'rate':>7}  verdict")
    for r in results:
        if "error" in r:
            print(f"{r['path']:<58}  ERROR {r['error'][:30]}")
            continue
        print(f"{Path(r['path']).name:<58} {r['with_xy']:>5} {r['leak']:>5} "
              f"{r['leak_rate']:>7.3f}  {r['verdict']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out}")
    return 0 if all(r.get("verdict") != "LEAKED" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
