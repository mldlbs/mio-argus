"""M1b 分析判据的单元测试（用合成计数，不需要训练）。"""
import pytest

import e1l_m1b_analyze as AN


def make_row(arm, H, succ, acc_t, prog, marks=None):
    return {"arm": arm, "mode": "zero" if arm == "A" else "mixed", "w": 24, "d": 1.0,
            "seed": 0, "rho": 0.3, "H": H, "n": 500, "succ": succ,
            "acc_by_t": [{"n": n, "ok": k} for n, k in acc_t],
            "by_progress": prog,
            "by_marks": marks or {"none": {"n": 4000, "ok": 2000}}}


def buckets(p0, p13, p47):
    return {"0": {"n": 500, "ok": p0}, "1-3": {"n": 1000, "ok": p13},
            "4-7": {"n": 1000, "ok": p47}}


def rows_effect():
    """B 明显更平：D1/D2/D3 都支持。"""
    a_t = [(500, 400), (500, 275), (500, 250), (500, 225),
           (500, 225), (500, 200), (500, 190), (500, 185)]
    b_t = [(500, 400), (500, 390), (500, 380), (500, 375),
           (500, 375), (500, 370), (500, 365), (500, 360)]
    return [make_row("A", 8, 150, a_t, buckets(400, 500, 450)),
            make_row("B", 8, 300, b_t, buckets(405, 750, 740)),
            make_row("A", 1, 400, [(500, 400)], buckets(400, 0, 0)),
            make_row("B", 1, 405, [(500, 405)], buckets(405, 0, 0)),
            make_row("A", 4, 300, [(500, 400), (500, 300), (500, 280), (500, 260)],
                     buckets(400, 500, 450)),
            make_row("B", 4, 380, [(500, 400), (500, 390), (500, 385), (500, 380)],
                     buckets(405, 750, 740))]


def rows_null():
    """两臂相同 → 不支持。"""
    t = [(500, 300)] * 8
    return [make_row("A", 8, 100, t, buckets(300, 300, 300)),
            make_row("B", 8, 100, t, buckets(300, 300, 300)),
            make_row("A", 1, 300, [(500, 300)], buckets(300, 0, 0)),
            make_row("B", 1, 300, [(500, 300)], buckets(300, 0, 0)),
            make_row("A", 4, 150, t, buckets(300, 300, 300)),
            make_row("B", 4, 150, t, buckets(300, 300, 300))]


def test_effect_supported():
    j = AN.judge(rows_effect())
    assert j["D1"] and j["D2"] and j["D3"]
    assert j["verdict"] == "机制得到初步支持"


def test_null_not_supported():
    j = AN.judge(rows_null())
    assert not j["D1"] and not j["D2"]
    assert j["verdict"] == "机制未获支持"


def test_partial_when_only_h8_improves():
    rows = rows_null()
    for r in rows:
        if r["arm"] == "B":
            r["succ"] = 300 if r["H"] == 8 else r["succ"]   # 只提 H8
    j = AN.judge(rows)
    assert j["D2"] and not j["D1"]
    assert j["verdict"] == "部分支持"


def test_d3_reversed_flagged():
    rows = rows_effect()
    for r in rows:
        if r["arm"] == "B":
            # 改善只出现在 progress=0
            r["by_progress"] = {"0": {"n": 500, "ok": 450},
                                "1-3": {"n": 1000, "ok": 300},
                                "4-7": {"n": 1000, "ok": 300}}
    j = AN.judge(rows)
    assert j["D3"] is False and j["D3_reversed"] is True


def test_missing_arm_raises():
    with pytest.raises(SystemExit):
        AN.judge([make_row("A", 8, 1, [(500, 1)] * 8, buckets(1, 1, 1))])
