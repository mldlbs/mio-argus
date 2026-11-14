"""看护状态面板：本地网页，随时查看 duck_guard 的运行状态。

用法：
    python experiments/computer_use/guard_panel.py

然后浏览器打开 http://127.0.0.1:8770 （会自动打开一次）

设计原则：
  - 只读为主：只读取 status/log/events/rules，不点游戏窗口；
  - 唯一写操作是"停止看护"（写 stop 标记文件，看护下一轮自己退出）；
  - 零依赖：只用标准库 http.server。
"""
from __future__ import annotations

import ctypes
import json
import re
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "computer_use"
PORT = 8770

STATUS = OUT / "duck_guard_status.json"
PIDF = OUT / "duck_guard.pid"
LOG = OUT / "duck_guard.log"
EVENTS = OUT / "duck_guard_events.jsonl"
STOPF = OUT / "duck_guard.stop"
CONFIG = HERE / "duck_guard_config.json"
RULES = OUT / "exp_loop" / "rules.json"


def _alive(pid: int) -> bool:
    """Windows 上判断进程是否存活（不需要 psutil）。"""
    if not pid:
        return False
    PROCESS_QUERY_LIMITED = 0x1000
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED, False, pid)
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return bool(ok) and code.value == 259      # STILL_ACTIVE


def _tail(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 200_000))
            data = fh.read().decode("utf-8", "ignore")
    except Exception:
        return []
    return data.splitlines()[-n:]


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _age_seconds(ts: str) -> int | None:
    """日志里的 [HH:MM:SS] → 距今多少秒。"""
    try:
        h, m, s = (int(x) for x in ts.split(":"))
    except Exception:
        return None
    now = datetime.now()
    t = now.replace(hour=h, minute=m, second=s, microsecond=0)
    diff = (now - t).total_seconds()
    if diff < -43200:            # 跨天
        diff += 86400
    return int(diff)


def build_state() -> dict:
    pid = 0
    if PIDF.exists():
        try:
            pid = int(PIDF.read_text(encoding="utf-8").strip())
        except Exception:
            pid = 0
    status = _json(STATUS)

    log_lines = _tail(LOG, 120)
    last_ts, last_msg = None, ""
    for line in reversed(log_lines):
        m = re.match(r"^\[(\d\d:\d\d:\d\d)\]\s*(.*)$", line)
        if m:
            last_ts, last_msg = m.group(1), m.group(2)
            break
    age = _age_seconds(last_ts) if last_ts else None

    # 当前层级 / 任务
    tier = task = None
    for line in reversed(log_lines):
        m = re.search(r"\[(routine|upgrade|edu|daily)\]\s*(\w+)", line)
        if m:
            tier, task = m.group(1), m.group(2)
            break

    events = []
    for line in _tail(EVENTS, 40):
        try:
            d = json.loads(line)
        except Exception:
            continue
        events.append({
            "ts": d.get("ts") or d.get("timestamp") or "",
            "action": d.get("action"),
            "detail": {k: v for k, v in d.items()
                       if k not in ("ts", "timestamp", "action")},
        })

    rules = [r for r in _json(RULES).get("rules", []) if r.get("status") == "active"]
    cfg = _json(CONFIG)
    tiers = (cfg.get("tiers") or {})

    return {
        "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pid": pid,
        "alive": _alive(pid),
        "stopping": STOPF.exists(),
        "last_log_ts": last_ts,
        "last_log_age": age,
        "last_log_msg": last_msg,
        "stalled": (age is not None and age > 600),
        "tier": tier,
        "task": task,
        "status": status,
        "log": log_lines[-40:],
        "events": events,
        "rules": [{"id": r.get("rule_id"), "statement": r.get("statement"),
                   "val": (r.get("metrics") or {}).get("val")} for r in rules],
        "tiers_cfg": {k: {"enabled": v.get("enabled"),
                          "interval": v.get("interval"),
                          "tasks": v.get("tasks")}
                      for k, v in tiers.items()},
        "actions": cfg.get("actions", {}),
    }


PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>DUCK 看护面板</title>
<style>
:root{--bg:#0f1216;--card:#171c22;--line:#242c36;--fg:#d7dee6;--dim:#7d8894;
--ok:#3ddc97;--warn:#ffcc4d;--bad:#ff5c5c;--accent:#4da3ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 "Segoe UI",system-ui,sans-serif;padding:18px}
h1{font-size:17px;margin:0 0 4px;letter-spacing:.5px}
.sub{color:var(--dim);font-size:12px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:12px;color:var(--dim);margin:0 0 8px;font-weight:600;
letter-spacing:.6px;text-transform:uppercase}
.big{font-size:22px;font-weight:600}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.row{display:flex;justify-content:space-between;padding:3px 0;
border-bottom:1px dashed #1e252d}
.row:last-child{border:0}
.k{color:var(--dim)}
pre{background:#0b0e12;border:1px solid var(--line);border-radius:8px;
padding:10px;max-height:300px;overflow:auto;font-size:12px;margin:0;
white-space:pre-wrap;word-break:break-all}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:4px 6px;border-bottom:1px solid #1e252d;text-align:left}
th{color:var(--dim);font-weight:600;font-size:12px}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;
border:1px solid var(--line)}
button{background:#222b35;color:var(--fg);border:1px solid var(--line);
border-radius:8px;padding:6px 12px;cursor:pointer}
button:hover{border-color:var(--accent)}
.bar{display:flex;gap:10px;align-items:center;margin-bottom:14px}
</style></head><body>
<h1>DUCK 看护面板</h1>
<div class="sub" id="sub">加载中…</div>
<div class="bar">
  <span class="pill" id="pill">—</span>
  <button onclick="cmd('stop')">停止看护</button>
  <button onclick="location.reload()">刷新</button>
</div>

<div class="grid">
  <div class="card"><h2>运行状态</h2>
    <div class="big" id="state">—</div>
    <div class="row"><span class="k">PID</span><span id="pid">—</span></div>
    <div class="row"><span class="k">最后活动</span><span id="age">—</span></div>
    <div class="row"><span class="k">当前</span><span id="cur">—</span></div>
  </div>
  <div class="card"><h2>本轮统计</h2>
    <div class="row"><span class="k">喂食</span><span id="feeds">—</span></div>
    <div class="row"><span class="k">合并</span><span id="merges">—</span></div>
    <div class="row"><span class="k">敲蛋</span><span id="cracks">—</span></div>
    <div class="row"><span class="k">领取/广告</span><span id="claims">—</span></div>
    <div class="row"><span class="k">错误</span><span id="errors">—</span></div>
  </div>
  <div class="card"><h2>资源</h2>
    <div class="row"><span class="k">玉米</span><span id="corn">—</span></div>
    <div class="row"><span class="k">星星</span><span id="stars">—</span></div>
    <div class="row"><span class="k">队列</span><span id="queue">—</span></div>
    <div class="row"><span class="k">页面</span><span id="page">—</span></div>
  </div>
  <div class="card"><h2>经验规则</h2>
    <div id="rules" style="font-size:12px"></div>
  </div>
</div>

<div class="grid" style="grid-template-columns:1.15fr .85fr">
  <div class="card"><h2>鸭子名单（最近一次走查）</h2>
    <table><thead><tr><th>#</th><th>状态</th></tr></thead>
    <tbody id="roster"></tbody></table>
  </div>
  <div class="card"><h2>调度</h2><div id="tiers" style="font-size:12px"></div></div>
</div>

<div class="grid" style="grid-template-columns:1fr 1fr">
  <div class="card"><h2>最近日志</h2><pre id="log"></pre></div>
  <div class="card"><h2>最近事件</h2><pre id="events"></pre></div>
</div>

<script>
async function cmd(what){
  if(what==='stop' && !confirm('确定停止看护？')) return;
  await fetch('/api/'+what,{method:'POST'});
  setTimeout(refresh, 600);
}
function fmt(d){
  document.getElementById('sub').textContent = '更新于 ' + d.now + '（每 2 秒自动刷新）';
  const s = d.status || {};
  document.getElementById('pid').textContent = d.pid || '—';
  document.getElementById('state').textContent = d.alive ? '运行中' : '未运行';
  document.getElementById('state').className = 'big ' + (d.alive ? 'ok' : 'bad');
  document.getElementById('pill').textContent = d.alive
    ? (d.stalled ? '⚠ 可能卡住（>' + Math.round(d.last_log_age/60) + ' 分钟无输出）' : '● 在线')
    : '○ 离线';
  document.getElementById('pill').className = 'pill ' + (d.alive ? (d.stalled?'warn':'ok') : 'bad');
  document.getElementById('age').textContent = d.last_log_ts
    ? d.last_log_ts + '（' + d.last_log_age + ' 秒前）' : '—';
  document.getElementById('cur').textContent = (d.tier? d.tier+' / ':'') + (d.task || '—');
  document.getElementById('feeds').textContent = s.feeds ?? '—';
  document.getElementById('merges').textContent = s.merges ?? '—';
  document.getElementById('cracks').textContent = s.cracks ?? '—';
  document.getElementById('claims').textContent = (s.claims ?? 0) + ' / ' + (s.ads_watched ?? 0);
  document.getElementById('errors').textContent = s.errors ?? '—';
  document.getElementById('corn').textContent = s.corn ?? '—';
  document.getElementById('stars').textContent = s.stars ?? '—';
  document.getElementById('queue').textContent = s.queue ?? '—';
  document.getElementById('page').textContent = s.page ?? '—';

  document.getElementById('rules').innerHTML = (d.rules||[]).map(r=>{
    const v = r.val||{};
    return '<div class="row"><span>'+r.id+'</span><span class="k">R='
      +(v.recall??'—')+' FPR='+(v.fpr??'—')+'</span></div>';
  }).join('') || '<span class="k">无</span>';

  document.getElementById('roster').innerHTML = (s.roster||[]).map((x,i)=>
    '<tr><td>'+(i+1)+'</td><td>'+x+'</td></tr>').join('')
    || '<tr><td colspan=2 class="k">无数据</td></tr>';

  document.getElementById('tiers').innerHTML = Object.entries(d.tiers_cfg||{}).map(
    ([k,v])=>'<div class="row"><span>'+k+'</span><span class="k">每 '+v.interval
    +'s · '+(v.tasks||[]).join(' → ')+'</span></div>').join('');

  document.getElementById('log').textContent = (d.log||[]).join('\n');
  document.getElementById('events').textContent = (d.events||[]).slice(-18)
    .map(e=>e.ts+'  '+e.action).join('\n');
}
async function refresh(){
  try{ fmt(await (await fetch('/api/state')).json()); }catch(e){}
}
refresh(); setInterval(refresh, 2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            self._send(200, json.dumps(build_state(), ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
        else:
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")

    def do_POST(self):
        if self.path.startswith("/api/stop"):
            STOPF.write_text("1", encoding="utf-8")
            self._send(200, b'{"ok":true}', "application/json")
        elif self.path.startswith("/api/start"):
            try:
                STOPF.unlink(missing_ok=True)
                subprocess.Popen(
                    [sys.executable, str(HERE / "duck_guard.py"), "--minutes", "0"],
                    cwd=str(ROOT), creationflags=0x00000008)  # DETACHED_PROCESS
                self._send(200, b'{"ok":true}', "application/json")
            except Exception as exc:
                self._send(500, json.dumps({"error": str(exc)}).encode(),
                           "application/json")
        else:
            self._send(404, b"{}", "application/json")


def _fmt_line(st: dict) -> str:
    s = st.get("status") or {}
    state = "运行中" if st["alive"] else "未运行"
    return (f"{st['now']}  {state}  PID={st['pid'] or '-'}  "
            f"最后活动 {st['last_log_age']}s 前  [{st.get('tier') or '-'}/"
            f"{st.get('task') or '-'}]  喂食{s.get('feeds', '-')} 合并{s.get('merges', '-')}"
            f" 星{s.get('stars', '-')}")


def run_tui(seconds: int = 0) -> int:
    """终端面板：每 2 秒原地刷新一次。"""
    t0 = time.time()
    try:
        while True:
            st = build_state()
            block = [
                "=" * 78,
                "DUCK 看护面板（终端）    " + st["now"],
                "=" * 78,
                _fmt_line(st),
                "-" * 78,
            ]
            s = st.get("status") or {}
            block.append(f"PID {st['pid'] or '-'} | 存活 {st['alive']} | "
                         f"卡住 {st['stalled']} | 停止标记 {st['stopping']}")
            block.append(f"当前任务: {st.get('tier') or '-'} / {st.get('task') or '-'}")
            block.append(f"统计: 喂食 {s.get('feeds')} 合并 {s.get('merges')} "
                         f"敲蛋 {s.get('cracks')} 领奖 {s.get('claims')} "
                         f"广告 {s.get('ads_watched')} 错误 {s.get('errors')}")
            block.append(f"资源: 玉米 {s.get('corn')} 星星 {s.get('stars')} "
                         f"队列 {s.get('queue')} 页面 {s.get('page')}")
            roster = s.get("roster") or []
            block.append(f"鸭子名单（{len(roster)}）:")
            for i, r in enumerate(roster[:12], 1):
                block.append(f"  {i:2d}. {r}")
            block.append(f"经验规则 active: {[r['id'] for r in st['rules']]}")
            block.append("-" * 78 + " 最近日志")
            block.extend(st["log"][-10:])
            print("\033[2J\033[H" + "\n".join(block), flush=True)
            if seconds and time.time() - t0 > seconds:
                return 0
            time.sleep(2)
    except KeyboardInterrupt:
        return 0


def run_win(seconds: int = 0) -> int:
    """桌面置顶小窗（Tkinter，标准库）。"""
    import tkinter as tk

    root = tk.Tk()
    root.title("DUCK 看护")
    root.configure(bg="#0f1216")
    root.geometry("360x300+40+40")
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.96)

    txt = tk.Label(root, text="加载中…", justify="left", anchor="nw",
                   bg="#0f1216", fg="#d7dee6",
                   font=("Consolas", 10), wraplength=340)
    txt.pack(fill="both", expand=True, padx=10, pady=(10, 4))

    bar = tk.Frame(root, bg="#0f1216")
    bar.pack(fill="x", padx=10, pady=(0, 10))

    def stop_guard():
        STOPF.write_text("1", encoding="utf-8")

    tk.Button(bar, text="停止看护", command=stop_guard,
              bg="#222b35", fg="#d7dee6", relief="flat").pack(side="left")
    tk.Button(bar, text="关闭窗口", command=root.destroy,
              bg="#222b35", fg="#d7dee6", relief="flat").pack(side="right")

    t0 = time.time()

    def tick():
        st = build_state()
        s = st.get("status") or {}
        dot = "● 运行中" if st["alive"] else "○ 未运行"
        if st["alive"] and st["stalled"]:
            dot = "⚠ 可能卡住"
        roster = s.get("roster") or []
        head = (f"{dot}   PID {st['pid'] or '-'}\n"
                f"最后活动：{st['last_log_age']} 秒前\n"
                f"当前：{st.get('tier') or '-'} / {st.get('task') or '-'}\n"
                f"喂食 {s.get('feeds', '-')}   合并 {s.get('merges', '-')}"
                f"   敲蛋 {s.get('cracks', '-')}\n"
                f"玉米 {s.get('corn', '-')}   星星 {s.get('stars', '-')}\n"
                f"鸭子 {len(roster)} 只\n"
                "-" * 30 + "\n")
        body = "\n".join(st["log"][-8:])
        txt.config(text=head + body)
        if seconds and time.time() - t0 > seconds:
            root.destroy()
            return
        root.after(2000, tick)

    tick()
    root.mainloop()
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="DUCK 看护状态面板")
    ap.add_argument("--mode", choices=("web", "tui", "win"), default="web")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--seconds", type=int, default=0,
                    help="自动退出时间（秒），0=一直运行")
    args = ap.parse_args()

    if args.mode == "tui":
        return run_tui(args.seconds)
    if args.mode == "win":
        return run_win(args.seconds)

    port = args.port
    for _ in range(20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            port += 1
    else:
        print("端口都被占用")
        return 1
    url = f"http://127.0.0.1:{port}"
    print(f"看护面板：{url}   (Ctrl+C 结束)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
