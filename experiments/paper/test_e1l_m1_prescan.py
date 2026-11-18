"""E1-L · M1 harness 的逻辑测试（不训练、不需要 GPU）。

对应审查条件：
 1/2  执行语义 + 误差累积  → test_error_marks_persist_and_change_next_frame
 3    H 参数同一套任务/模型 → test_prefix_invariance_across_H
 4    统计口径              → test_wilson_ci + test_success_requires_all_steps
 5    随机性隔离/可复现      → test_reproducible / test_geometry_independent_of_decoys
 6    门禁用实测值           → test_rows_consumed_by_band_gate
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import e1l_m1_prescan as M
import p3_env as P3

CFG = M.EnvCfg6()


def frame(**kw):
    args = dict(cfg=CFG, labels=M.fresh_labels(CFG, np.random.default_rng(3)),
                progress=0, H=4, wrong=[False] * 36, done=[False] * 36,
                target_label=None, d=0.0, level="flat", decoy_seed=0, geom_seed=7)
    args.update(kw)
    if args["target_label"] is None:
        args["target_label"] = args["labels"][0]
    return M.render_state(**args)


def test_flat_zero_decoys_matches_p3_render_ui():
    """契约：flat + d=0 必须与 p3_env.render_ui 逐像素相同（防止复制渲染时漂移）。"""
    labels = M.fresh_labels(CFG, np.random.default_rng(11))
    wrong = [False] * 36
    wrong[5] = True
    done = [False] * 36
    done[6] = True
    for progress in (0, 2):
        a, _ = P3.render_ui(labels, progress, 4, wrong, done, CFG, geom_seed=123)
        b, _ = M.render_state(CFG, labels, progress, 4, wrong, done, labels[0],
                              d=0.0, level="flat", decoy_seed=9, geom_seed=123,
                              bar_total=4)
        assert np.array_equal(a, b)


def test_error_marks_persist_and_change_next_frame():
    """错一步之后：进度不前进、红框留在屏幕上、下一步的输入确实变了。"""
    script = M.build_scripts(CFG, seed=0, rho=0.0, n_tasks=1)[0]

    good = M.rollout(M.OraclePolicy(), CFG, script, 4, 1.0, 0, 0, 0.0, keep_frames=True)
    bad = M.rollout(M.OraclePolicy(wrong_steps={0}), CFG, script, 4, 1.0, 0, 0, 0.0,
                    keep_frames=True)

    assert good["progress"] == 4 and good["success"]
    assert bad["progress"] == 3 and not bad["success"]
    assert bad["step_ok"][0] is False
    # 第 1 步的输入不同（红框 vs 灰色已完成的进度）
    assert not np.array_equal(good["frames"][1], bad["frames"][1])
    # 前进步数与 step_ok 一致
    assert sum(bad["step_ok"]) == bad["progress"]


def test_success_requires_all_steps():
    """任意一步错 → 整个 episode 失败；全对 → 每个 H 都成功。"""
    script = M.build_scripts(CFG, seed=1, rho=0.0, n_tasks=1)[0]
    for H in (1, 2, 4, 8):
        ok = M.rollout(M.OraclePolicy(), CFG, script, H, 0.0, 1, 0, 0.0)
        assert ok["success"] and ok["progress"] == H
        for k in range(H):
            r = M.rollout(M.OraclePolicy(wrong_steps={k}), CFG, script, H, 0.0, 1, 0, 0.0)
            assert not r["success"]
            assert r["progress"] == H - 1


@pytest.mark.parametrize("rho", [0.0, 0.3])
def test_prefix_invariance_across_H(rho):
    """同一任务：H=4 的轨迹必须是 H=8 的前 4 步（逐步逐像素相同）。"""
    script = M.build_scripts(CFG, seed=2, rho=rho, n_tasks=1)[0]
    a = M.rollout(M.OraclePolicy(), CFG, script, 4, 1.0, 2, 0, rho, keep_frames=True)
    b = M.rollout(M.OraclePolicy(), CFG, script, 8, 1.0, 2, 0, rho, keep_frames=True)
    assert a["actions"] == b["actions"][:4]
    for t in range(4):
        assert np.array_equal(a["frames"][t], b["frames"][t]), f"step {t} 不一致"


def test_reproducible():
    """同参数两次 rollout 完全一致（含动作与逐 step 结果）。"""
    script = M.build_scripts(CFG, seed=3, rho=0.3, n_tasks=1)[0]
    a = M.rollout(M.OraclePolicy(wrong_steps={1}), CFG, script, 4, 1.0, 3, 0, 0.3,
                  keep_frames=True)
    b = M.rollout(M.OraclePolicy(wrong_steps={1}), CFG, script, 4, 1.0, 3, 0, 0.3,
                  keep_frames=True)
    assert a["actions"] == b["actions"] and a["step_ok"] == b["step_ok"]
    for t in range(4):
        assert np.array_equal(a["frames"][t], b["frames"][t])


def test_geometry_independent_of_decoys():
    """干扰项只影响背景层：元素框的位置必须与 d=0 时完全相同。"""
    labels = M.fresh_labels(CFG, np.random.default_rng(5))
    _, els0 = M.render_state(CFG, labels, 0, 4, [False] * 36, [False] * 36, labels[0],
                             d=0.0, level="flat", decoy_seed=1, geom_seed=42)
    _, els1 = M.render_state(CFG, labels, 0, 4, [False] * 36, [False] * 36, labels[0],
                             d=1.0, level="flat", decoy_seed=1, geom_seed=42)
    assert [e[1] for e in els0] == [e[1] for e in els1]


def test_decoys_exclude_target_color_and_avoid_bar():
    """干扰项：不复制目标色；数量符合 d；不入侵进度条区域。"""
    labels = M.fresh_labels(CFG, np.random.default_rng(6))
    target = labels[3]
    tgt_color = tuple(int(v) for v in M.PALETTE[M.LABELS.index(target) % len(M.PALETTE)])
    top = M.BAR_Y1 + 4

    for d, expect in ((1.0, 36), (0.5, 18)):
        dec = M.gen_decoys(CFG, tgt_color, d, np.random.default_rng(0), top)
        assert len(dec) == expect, (d, len(dec))
        assert all(c != tgt_color for _, c in dec)
        assert all(b[1] >= top for b, _ in dec)
        assert all(b[2] < M.IMG_SIZE and b[3] < M.IMG_SIZE for b, _ in dec)


def test_wilson_ci():
    lo, hi = M.wilson_ci(0, 10)
    assert lo == 0.0 and abs(hi - 0.2776) < 1e-3
    lo, hi = M.wilson_ci(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3
    assert M.wilson_ci(0, 0) == (None, None)


def test_stats_shape_and_ci_cover_rate():
    """统计口径：逐 (候选, seed, H) 必须有 n / success / rate / CI，且 CI 覆盖 rate。"""
    scripts = M.build_scripts(CFG, seed=5, rho=0.0, n_tasks=4)
    allok = M.evaluate(M.OraclePolicy(), CFG, scripts, 2, 0.0, 5, 0.0)
    assert allok["n"] == 4 and allok["success"] == 4 and allok["rate"] == 1.0
    assert allok["ci"][0] <= allok["rate"] <= allok["ci"][1]
    assert len(allok["acc_by_step"]) == 2

    never = M.evaluate(M.OraclePolicy(wrong_steps={0}), CFG, scripts, 2, 0.0, 5, 0.0)
    assert never["success"] == 0 and never["rate"] == 0.0
    assert never["ci"][0] == 0.0 and never["ci"][1] > 0.0


def test_progress_is_cumulative_done_is_per_layout():
    """状态机口径：progress 累计不重置；wrong/done 是"当前布局"的标记，随 ρ 清空。"""
    script = M.build_scripts(CFG, seed=7, rho=1.0, n_tasks=1)[0]
    r = M.rollout(M.OraclePolicy(), CFG, script, 8, 0.0, 7, 0, 1.0, keep_states=True)
    assert r["success"] and r["progress"] == 8
    for t, (prog, wrong, done) in enumerate(r["states"]):
        assert prog == t, f"step {t}: progress 应为累计值 {t}，实际 {prog}"
        assert not any(done), f"step {t}: 漂移后 done 必须清空"
        assert not any(wrong), f"step {t}: 全对时不应有 wrong"

    # ρ=0：错一步后，标记保留到下一步的输入里；progress 只加正确的步
    script2 = M.build_scripts(CFG, seed=7, rho=0.0, n_tasks=1)[0]
    r2 = M.rollout(M.OraclePolicy(wrong_steps={1}), CFG, script2, 4, 0.0, 7, 0, 0.0,
                   keep_states=True)
    assert r2["progress"] == 3
    wrong_slot = (script2[1].target_slot + 1) % 36
    prog_at_2, wrong_at_2, done_at_2 = r2["states"][2]
    assert prog_at_2 == 1
    assert wrong_at_2[wrong_slot] is True
    assert done_at_2[script2[0].target_slot] is True


def test_eval_order_independence():
    """H 之间的结果与评估顺序无关（随机流只由 seed/task/t 决定）。"""
    scripts = M.build_scripts(CFG, seed=8, rho=0.3, n_tasks=5)
    a4 = M.evaluate(M.OraclePolicy(wrong_steps={2}), CFG, scripts, 4, 0.0, 8, 0.3)
    a8 = M.evaluate(M.OraclePolicy(wrong_steps={2}), CFG, scripts, 8, 0.0, 8, 0.3)
    b8 = M.evaluate(M.OraclePolicy(wrong_steps={2}), CFG, scripts, 8, 0.0, 8, 0.3)
    b4 = M.evaluate(M.OraclePolicy(wrong_steps={2}), CFG, scripts, 4, 0.0, 8, 0.3)
    assert (a4["success"], a8["success"]) == (b4["success"], b8["success"])
    assert a4["acc_by_step"] == b4["acc_by_step"]


def test_matrix_covers_both_rhos_without_extra_training():
    """执行矩阵：96 个评估格 = 4 候选 × 3 seed × 2 ρ × 4 H；训练只有 12 次。"""
    mx = M.expand_matrix(M.CANDS, [0, 1, 2], [0.3, 0.0])
    assert len(mx) == 96
    assert len(M.train_units(mx)) == 12
    tags = {m["tag"] for m in mx}
    assert tags == {f"{c[0]}@r{r}" for c in M.CANDS for r in (0.3, 0.0)}
    for t in tags:
        assert sum(1 for m in mx if m["tag"] == t) == 12
    assert all(m["tag"].endswith(("@r0.3", "@r0.0")) for m in mx)


def test_completeness_detects_missing_row():
    mx = M.expand_matrix(M.CANDS[:1], [0], [0.3, 0.0])
    rows = [{"cond": m["tag"], "seed": m["seed"], "x": m["H"]} for m in mx]
    assert M.check_completeness(rows, mx) == []
    assert len(M.check_completeness(rows[:-1], mx)) == 1


def test_rows_consumed_by_band_gate(tmp_path):
    """统计输出能直接被 prescan_gate 的 B 模式消费（门禁用实测值，不用预测）。"""
    script = M.build_scripts(CFG, seed=4, rho=0.0, n_tasks=2)
    rows = []
    for H in (1, 2):
        r = M.evaluate(M.OraclePolicy(), CFG, script, H, 0.0, 4, 0.0)
        rows.append({"cond": "cand@r0.0", "x": H, "seed": 0, "value": round(r["rate"], 4)})
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    here = Path(__file__).parent
    out = subprocess.run(
        [sys.executable, str(here / "prescan_gate.py"), "--runs", str(p),
         "--band", "0.2,0.9", "--monotone", "--monotone-dir", "down"],
        capture_output=True, text=True, encoding="utf-8", cwd=str(here))
    # 全对的替身策略 rate=1.0 → 顶出 [0.2,0.9] 上限，门禁必须拦下（exit 1）
    assert out.returncode == 1, out.stdout[-400:]
    assert "FAIL" in out.stdout


def test_gate_monotone_direction(tmp_path):
    """H 曲线应下降：用 down 判通过，用 up 判违规（防止把方向搞反）。"""
    rows = [{"cond": "c@r0.3", "x": H, "seed": 0, "value": v}
            for H, v in ((1, 0.85), (2, 0.72), (4, 0.52), (8, 0.27))]
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    here = Path(__file__).parent
    down = subprocess.run(
        [sys.executable, str(here / "prescan_gate.py"), "--runs", str(p),
         "--band", "0.2,0.9", "--monotone", "--monotone-dir", "down"],
        capture_output=True, text=True, encoding="utf-8", cwd=str(here))
    up = subprocess.run(
        [sys.executable, str(here / "prescan_gate.py"), "--runs", str(p),
         "--band", "0.2,0.9", "--monotone", "--monotone-dir", "up"],
        capture_output=True, text=True, encoding="utf-8", cwd=str(here))
    assert down.returncode == 0, down.stdout[-400:]
    assert up.returncode == 1, up.stdout[-400:]
