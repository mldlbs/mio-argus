"""稳健采集 Bigemap：清理残留 -> 启动 -> 轮询窗口与控件树 -> 截图+落盘。"""
import json
import subprocess
import time
from pathlib import Path

import pyautogui
from pywinauto import Desktop

ROOT = Path(r"D:\work\code\nanoGPT")
EXE = r"F:\Program Files (x86)\Bigemap GIS Office\MapDownloader.exe"
OUT = ROOT / "artifacts" / "uia_capture" / "bigemap"
LOG = Path(r"C:\Users\gf191\AppData\Local\Temp\opencode\bigemap_capture.log")
lines = []


def log(s):
    lines.append(str(s))
    LOG.write_text("\n".join(lines), encoding="utf-8")


def kill_all():
    subprocess.run(["taskkill", "/F", "/IM", "MapDownloader.exe"],
                   capture_output=True)


def find_window():
    for w in Desktop(backend="uia").windows(visible_only=True):
        try:
            if "Bigemap" in (w.window_text() or ""):
                return w
        except Exception:
            continue
    return None


def dismiss_running_dialog():
    for w in Desktop(backend="uia").windows(visible_only=True):
        try:
            t = w.window_text() or ""
            if t.strip() == "提示":
                for b in w.descendants(control_type="Button"):
                    if "OK" in (b.window_text() or "").upper():
                        b.click_input()
                        return True
        except Exception:
            continue
    return False


def capture(win, tag):
    r = win.rectangle()
    OUT.mkdir(parents=True, exist_ok=True)
    pyautogui.screenshot(region=(int(r.left), int(r.top),
                                 max(1, int(r.width())), max(1, int(r.height())))
                         ).save(OUT / f"shot_{tag}.png")
    nodes = []
    for c in win.descendants():
        try:
            ct = c.element_info.control_type
            name = (c.window_text() or "").strip()
            rr = c.rectangle()
            nodes.append({"type": ct, "name": name,
                          "class": c.element_info.class_name or "",
                          "automation_id": c.element_info.auto_id or "",
                          "rect": [int(rr.left), int(rr.top), int(rr.width()), int(rr.height())],
                          "enabled": bool(c.is_enabled())})
        except Exception:
            continue
    payload = {"title": win.window_text(), "window_rect": [int(r.left), int(r.top),
                                                           int(r.width()), int(r.height())],
               "n_nodes": len(nodes), "nodes": nodes}
    (OUT / f"tree_{tag}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    named = sum(1 for n in nodes if n["name"])
    log(f"[{tag}] nodes={len(nodes)} named={named}")
    return len(nodes), named


def main():
    for attempt in range(1, 4):
        log(f"=== attempt {attempt} ===")
        kill_all()
        time.sleep(2)
        subprocess.Popen([EXE])
        win = None
        deadline = time.time() + 120
        while time.time() < deadline:
            time.sleep(2)
            if dismiss_running_dialog():
                log("dismissed 'already running' dialog")
            win = find_window()
            if win is not None:
                n, named = capture(win, f"a{attempt}_t{int(time.time()) % 10000}")
                if n > 20:
                    log(f"success: {n} nodes")
                    print(f"ok nodes={n} named={named}")
                    return 0
                log(f"window found but only {n} nodes; keep waiting")
        log("timeout waiting for a usable window")
    print("failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
