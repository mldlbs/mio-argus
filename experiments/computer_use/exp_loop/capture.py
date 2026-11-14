"""采集原始观测 episode（不含结论）。

关键原则：**标签由独立判定器给出，不由采集方式给出。**
  - 点击类：用生产环境已在用的「高亮校验」当选址真值——
    选中格是否落在被点的位置（±0.022）。满足=成功(negative)，不满足=失败(positive)。
  - 识别类：用两个通道互相验证——仅识别通道返回空、而整页识别能读出文字，
    才算「识别通道失败」(positive)；两边都空=确实没内容(negative)。

这样"点缝隙却碰巧点中"会自动变成负样本，不再污染数据。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cv2
import pyautogui

import duck_guard as dg
import ui_icons
from episode import Episode, Frame, write_episodes

OUT = Path("results") / "computer_use" / "exp_loop"
FRAMES = OUT / "frames"


def _page_text(g) -> str:
    return " ".join(l["text"] for l in g.lines())


def _snap(g, ep_id: str, tag: str) -> str:
    FRAMES.mkdir(parents=True, exist_ok=True)
    img = ui_icons.grab(g.hwnd)
    rel = f"frames/{ep_id}_{tag}.png"
    cv2.imwrite(str(OUT / rel), img)
    return rel


def _ensure_duck(g, tries: int = 6) -> bool:
    """页面就绪门禁：必须是鸭子页且槽位带可见，否则这批数据就是垃圾。"""
    for _ in range(tries):
        # 以「槽位带能否检出图标」为主判据：OCR 的页面标签在动画期会抖
        if len(dg.detect_slot_icons(g)) >= 3:
            return True
        g.close_popup()
        dg.go_home(g)
        dg.goto(g, "duck", "duck", tries=2)
        time.sleep(1.2)
    return False


def _selection_verified(g, x: float, tol: float = 0.022) -> bool:
    """独立判定：**仍在鸭子页**，且选中格落在被点的位置。

    必须带页面检查——点挂锁会跳到「购买槽位」页，那页上也可能测到亮块，
    否则会把"点错导航"误标成成功（实测发生过）。
    """
    if dg.current_page(g, tries=1)[0] != "duck":
        return False
    icons = dg.detect_slot_icons(g)
    sel = dg.selected_icon(icons)
    return sel is not None and abs(icons[sel]["x"] - x) < tol


def _click_episode(g, ep_id: str, x: float, origin: str,
                   text_target: str = "", expected="selection_change") -> Episode:
    before_txt = _page_text(g)
    before_f = _snap(g, ep_id, "before")
    pyautogui.click(g.l + int(x * g.w), g.t + int(dg.SLOT_CY * g.h))
    time.sleep(1.2)
    ok = _selection_verified(g, x)
    after_txt = _page_text(g)
    after_f = _snap(g, ep_id, "after")
    ep = Episode(
        id=ep_id, origin=origin,
        action={"type": "click", "x": round(x, 3), "y": dg.SLOT_CY,
                "text_target": text_target},
        before=Frame(frame=before_f, text=before_txt, ocr_channel="full"),
        after=Frame(frame=after_f, text=after_txt, ocr_channel="full"),
        expected={"kind": expected},
    )
    ep.extra["verified"] = ok          # 真值存在 extra 里，归因不读
    return ep


def capture(dst: Path = OUT) -> list[Episode]:
    global OUT, FRAMES
    OUT = dst
    FRAMES = dst / "frames"

    h = dg.find_game_window()
    g = dg.Game(h)
    g.pin()
    eps: list[Episode] = []
    labels: dict[str, str] = {}

    stamp = str(int(time.time()) % 100000)
    def add(ep: Episode) -> None:
        ep.id = f"{ep.id}_{stamp}"
        eps.append(ep)
        labels[ep.id] = "negative" if ep.extra.get("verified") else "positive"

    # ---------------- 点击类：缝隙
    if not _ensure_duck(g):
        print("鸭子页未就绪，采集中止")
        return []
    time.sleep(0.8)
    icons = dg.detect_slot_icons(g)
    gaps = []
    for i in range(len(icons) - 1):
        a, b = icons[i]["x"], icons[i + 1]["x"]
        if 0.20 < (a + b) / 2 < 0.95:
            gaps.append(round((a + b) / 2, 3))
    for k, gx in enumerate(gaps[:10]):
        add(_click_episode(g, f"ep_gap_{k + 1:02d}", gx, "gap_click"))

    # ---------------- 点击类：长句说明文字
    g.nav("task", wait=2.0)
    time.sleep(1.2)
    if dg._open_ads_tab(g):
        long_lines = [l for l in g.lines()
                      if len(l["text"].strip()) >= 16
                      and "DUCK × MY" not in l["text"]
                      and 0.25 < (l["rect"][1] - g.t) / g.h < 0.80]
        for k, line in enumerate(long_lines[:5]):
            r = line["rect"]
            add(_click_episode(g, f"ep_sent_{k + 1:02d}",
                               (r[0] - g.l) / g.w, "sentence_click",
                               text_target=line["text"].strip(),
                               expected="text_change"))

    # ---------------- 点击类：点错导航（挂锁区 → 购买页）
    if not _ensure_duck(g):
        print("鸭子页未就绪（导航采集）→ 跳过该段")
        pass
    time.sleep(0.8)
    for k in range(6):
        icons = dg.detect_slot_icons(g)
        if len(icons) < 3:
            break
        target = icons[-2]
        add(_click_episode(g, f"ep_nav_{k + 1:02d}", target["x"], "wrong_nav"))
        if dg.current_page(g, tries=1)[0] != "duck":
            g.close_popup()
        dg.goto(g, "duck", "duck", tries=2)
        time.sleep(1.0)

    # ---------------- 识别类：仅识别通道 vs 整页识别 互相验证
    import bigemap_runner as bench

    def _ocr_pair(crop):
        big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        o1 = bench.ocr_engine()(big, use_det=False, use_cls=False)
        r1 = o1[0] if isinstance(o1, tuple) else o1
        fast = " ".join(str(i[0]) for i in (r1 or []) if isinstance(i, (list, tuple)))
        o2 = bench.ocr_engine()(big)
        r2 = o2[0] if isinstance(o2, tuple) else o2
        full = " ".join(str(i[1] if len(i) == 3 else i[0]) for i in (r2 or []))
        return fast.strip(), full.strip()

    def _ocr_episode(ep_id, crop, origin):
        FRAMES.mkdir(parents=True, exist_ok=True)
        rel = f"frames/{ep_id}_crop.png"
        cv2.imwrite(str(OUT / rel), crop)
        fast, full = _ocr_pair(crop)
        ep = Episode(
            id=ep_id, origin=origin,
            action={"type": "recognize", "channel": "fast"},
            before=Frame(frame=rel, text="", ocr_channel="fast"),
            after=Frame(frame=rel, text=fast, ocr_channel="fast"),
            expected={"kind": "ocr_text"},
        )
        ep.extra["full_text"] = full
        # 真值：仅识别空 且 整页识别有字 → 识别通道失败
        ep.extra["verified"] = not (fast == "" and full != "")
        add(ep)

    dg.go_home(g)
    dg.goto(g, "duck", "duck", tries=3)
    time.sleep(1.2)
    for k in range(8):
        dg._step_duck(g, +1)
        time.sleep(0.4)
        img = ui_icons.grab(g.hwnd)
        H, W = img.shape[:2]
        rx, ry, rw, rh = dg.FEED_BAND
        crop = img[int(ry * H):int((ry + rh) * H), int(rx * W):int((rx + rw) * W)]
        if crop.size:
            _ocr_episode(f"ep_ocr_{k + 1:02d}", crop, "empty_ocr")

    img = ui_icons.grab(g.hwnd)
    H, W = img.shape[:2]
    blank = img[int(0.855 * H):int(0.880 * H), int(0.10 * W):int(0.42 * W)]
    if blank.size:
        for k in range(8):
            _ocr_episode(f"ep_blank_{k + 1:02d}", blank, "blank_ocr")

    # ---------------- 点击类：正常选中（负样本主力）
    if not _ensure_duck(g):
        print("鸭子页未就绪（正常点击采集）→ 跳过该段")
        pass
    time.sleep(0.8)
    for k in range(24):
        icons = dg.detect_slot_icons(g)
        sel = dg.selected_icon(icons)
        if sel is None:
            g.nav("duck", wait=1.2)
            time.sleep(0.8)
            continue
        pitch = dg.slot_pitch(icons)
        nxt = round(icons[sel]["x"] + (pitch if k % 2 == 0 else -pitch), 3)
        if not (0.20 <= nxt <= 0.95):
            g.nav("duck", wait=1.2)
            time.sleep(0.8)
            continue
        add(_click_episode(g, f"ep_good_{k + 1:02d}", nxt, "good_click"))

    # ---------------- 证伪块：右侧区域的**合法**点击（负样本）
    # 自动合成可能会提出「点击 x 靠右 = 高风险」这种位置规则，
    # 那多半是采集几何的假象。这里专门采右侧合法点击来证伪它。
    if _ensure_duck(g):
        for k in range(12):
            icons = dg.detect_slot_icons(g)
            sel = dg.selected_icon(icons)
            if sel is None:
                g.nav("duck", wait=1.2)
                time.sleep(0.8)
                continue
            # 往右挪到 x>0.55 再点
            ok = True
            for _ in range(6):
                icons = dg.detect_slot_icons(g)
                sel = dg.selected_icon(icons)
                if sel is None or icons[sel]["x"] > 0.55:
                    break
                ok, _nx = dg._step_duck(g, +1)
                time.sleep(0.3)
                if not ok:
                    break
            icons = dg.detect_slot_icons(g)
            sel = dg.selected_icon(icons)
            if sel is None or icons[sel]["x"] <= 0.55:
                g.nav("duck", wait=1.2)
                time.sleep(0.8)
                continue
            add(_click_episode(g, f"ep_right_{k + 1:02d}", icons[sel]["x"],
                               "right_click"))

    write_episodes(OUT / "episodes_batch.jsonl", eps)
    with (OUT / "labels_batch.jsonl").open("w", encoding="utf-8") as fh:
        for k, v in labels.items():
            fh.write(json.dumps({"id": k, "label": v}, ensure_ascii=False) + "\n")
    pos = sum(v == "positive" for v in labels.values())
    print(f"episodes={len(eps)} positives={pos} negatives={len(labels) - pos}")
    return eps


if __name__ == "__main__":
    capture()
