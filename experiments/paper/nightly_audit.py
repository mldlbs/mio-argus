"""L0 夜间体检：只做可机检、可判真假的事，FAIL 才值得人看。

设计原则（见 docs/RESEARCH_PORTFOLIO.md）：
  - 不调模型、不占 GPU、不联网、不写代码；
  - 每项检查是二元的 PASS/FAIL/SKIP；
  - 事实追加进 results/paper/LEDGER.json（按签名去重，不重复记同一条）；
  - 任一 FAIL → 退出码 1（供 automation 判断是否升级）。

用法：
    python experiments/paper/nightly_audit.py            # 全量检查
    python experiments/paper/nightly_audit.py --fast     # 跳过 pytest
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "results/paper/LEDGER.json"
REPORT = ROOT / "results/paper/NIGHTLY_AUDIT_LATEST.md"
CKPT_DIR = ROOT / "artifacts/checkpoints"
G6_DIR = ROOT / "results/computer_use"


def sh(args, timeout=600):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# ------------------------------------------------------------------ checks ---
def check_git():
    _, dirty = sh(["git", "status", "--porcelain"])
    _, ahead = sh(["git", "rev-list", "--count", "origin/master..HEAD"])
    n_dirty = len([l for l in dirty.splitlines() if l.strip()])
    n_ahead = int(ahead.strip() or 0)
    status = "PASS" if n_ahead == 0 else "FAIL"
    return {"id": "git_sync", "status": status,
            "detail": f"未提交={n_dirty} 未推送={n_ahead}",
            "fact": f"unpushed={n_ahead}" if n_ahead else None}


def check_leak():
    """新数据集是否把坐标写进了指令。"""
    code, out = sh([sys.executable, "experiments/computer_use/audit_data_leak.py",
                    "--out", "results/computer_use/data_leak_audit.json"])
    leaked = re.findall(r"^(\S+\.json)\s+\d+\s+\d+\s+1\.000\s+LEAKED", out, re.M)
    return {"id": "data_leak", "status": "PASS" if not leaked else "FAIL",
            "detail": f"leaked={leaked}" if leaked else "无新泄露",
            "fact": f"leaked dataset: {','.join(leaked)}" if leaked else None}


def check_g6():
    """每个 checkpoint 必须有对应的 G6 记录（模态消融 json）。"""
    if not CKPT_DIR.exists():
        return {"id": "g6_coverage", "status": "SKIP", "detail": "无 checkpoint 目录",
                "fact": None}
    ckpts = sorted(p.stem for p in CKPT_DIR.glob("*.pth"))
    covered, missing = [], []
    for stem in ckpts:
        hits = list(G6_DIR.glob(f"*{stem}*.json")) + list(G6_DIR.glob(f"modality_ablation_{stem}.json"))
        (covered if hits else missing).append(stem)
    status = "PASS" if not missing else "FAIL"
    return {"id": "g6_coverage", "status": status,
            "detail": f"{len(covered)}/{len(ckpts)} 有 G6 记录",
            "fact": (f"checkpoints without G6: {','.join(missing[:8])}" if missing else None),
            "missing": missing}


def check_tests():
    code, out = sh([sys.executable, "-m", "pytest", "-q"], timeout=900)
    m = re.search(r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) error)?", out)
    if not m:
        return {"id": "tests", "status": "SKIP", "detail": "无法解析 pytest 输出",
                "fact": None}
    counts = {"passed": int(m.group(1)), "failed": int(m.group(2) or 0),
              "errors": int(m.group(3) or 0)}
    base_path = ROOT / "results/paper/TEST_BASELINE.json"
    if not base_path.exists():
        base_path.write_text(json.dumps(counts, indent=1), encoding="utf-8")
        return {"id": "tests", "status": "SKIP",
                "detail": f"建立基线 {counts}", "fact": None, "counts": counts}
    base = json.loads(base_path.read_text(encoding="utf-8"))
    bad_now = counts["failed"] + counts["errors"]
    bad_base = base.get("failed", 0) + base.get("errors", 0)
    status = "PASS" if bad_now <= bad_base else "FAIL"
    return {"id": "tests", "status": status,
            "detail": f"passed={counts['passed']} failed={counts['failed']} "
                      f"error={counts['errors']}（基线 {base.get('failed',0)}/{base.get('errors',0)}）",
            "fact": (f"tests regressed: failed+error {bad_base}->{bad_now}" if status == "FAIL"
                     else None), "counts": counts}


def check_ledger_age(ledger):
    if not ledger["facts"]:
        return {"id": "ledger_fresh", "status": "SKIP", "detail": "账本为空", "fact": None}
    last = max(f["ts"] for f in ledger["facts"])
    days = (time.time() - last) / 86400
    return {"id": "ledger_fresh",
            "status": "PASS" if days < 7 else "FAIL",
            "detail": f"最后一条事实 {days:.1f} 天前",
            "fact": f"ledger stale {days:.1f} days" if days >= 7 else None}


# ------------------------------------------------------------------ ledger ---
def load_ledger():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"version": 1, "facts": []}


def append_facts(ledger, checks):
    known = {f["sig"] for f in ledger["facts"]}
    added = 0
    for c in checks:
        if not c.get("fact"):
            continue
        sig = hashlib.sha1(c["fact"].encode("utf-8")).hexdigest()[:16]
        if sig in known:
            continue
        ledger["facts"].append({"id": f"f{len(ledger['facts'])+1:04d}", "sig": sig,
                                "check": c["id"], "fact": c["fact"], "ts": time.time(),
                                "date": time.strftime("%Y-%m-%d %H:%M")})
        known.add(sig)
        added += 1
    return added


def load_acks():
    """已确认的事实（避免同一个 FAIL 每天刷屏）。格式：
    {"acked": [{"check": "data_leak", "match": "子串", "reason": "...", "date": "..."}]}
    """
    p = ROOT / "results/paper/LEDGER_ACK.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("acked", [])
    except Exception:                                        # noqa: BLE001
        return []


def apply_acks(checks, acks):
    for c in checks:
        if c["status"] != "FAIL" or not c.get("fact"):
            continue
        for a in acks:
            if a.get("check") == c["id"] and a.get("match", "\0") in c["fact"]:
                c["status"] = "ACK"
                c["detail"] = f"{c['detail']}（已确认：{a.get('reason', '')[:40]}）"
                break
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="跳过 pytest")
    args = ap.parse_args()

    ledger = load_ledger()
    checks = [check_git(), check_leak(), check_g6(), check_ledger_age(ledger)]
    if not args.fast:
        checks.append(check_tests())
    checks = apply_acks(checks, load_acks())

    added = append_facts(ledger, checks)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8")

    fails = [c for c in checks if c["status"] == "FAIL"]
    lines = [f"# 夜间体检 {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"检查 {len(checks)} 项，FAIL {len(fails)} 项，新增事实 {added} 条", "",
             "| 检查 | 结果 | 详情 |", "|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['id']} | {c['status']} | {c['detail']} |")
    if fails:
        lines += ["", "## 需要人看", ""] + [f"- **{c['id']}**: {c['detail']}" for c in fails]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[ledger] {len(ledger['facts'])} 条事实（本次 +{added}）")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
