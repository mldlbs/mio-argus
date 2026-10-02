"""
QGIS 桥客户端：truth / structure / act / shot

用法（从仓库根运行）：
    python experiments/qgis/qgis_bridge/client.py ping
    python experiments/qgis/qgis_bridge/client.py truth
    python experiments/qgis/qgis_bridge/client.py structure --limit 20
    python experiments/qgis/qgis_bridge/client.py act-json "{\"kind\":\"action\",\"text\":\"Zoom In\"}"
    python experiments/qgis/qgis_bridge/client.py shot experiments/qgis/qgis_bridge/_shot.png
"""
import argparse
import json
import os
import socket
import sys

PORT = int(os.environ.get("QGIS_BRIDGE_PORT", "8765"))


def call(req, port=PORT, timeout=60):
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    with s:
        f = s.makefile("rwb")
        f.write((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
        f.flush()
        line = f.readline()
        if not line:
            raise RuntimeError("bridge closed connection")
        return json.loads(line.decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ping", "truth", "structure", "act-json", "shot",
                                    "action", "click", "canvas", "windows", "win", "reset",
                                    "set-extent"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("arg2", nargs="?")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--maximize", action="store_true")
    args = ap.parse_args()

    if args.cmd == "reset":
        print(json.dumps(call({"op": "reset"}), indent=2, ensure_ascii=False))
    elif args.cmd == "set-extent":
        box = [float(v) for v in (args.arg or "").split(",")]
        print(json.dumps(call({"op": "set_extent", "box": box}),
                         indent=2, ensure_ascii=False))
    elif args.cmd == "win":
        req = {"op": "win", "maximize": args.maximize}
        if args.arg:
            req["w"] = int(args.arg)
        if args.arg2:
            req["h"] = int(args.arg2)
        print(json.dumps(call(req), indent=2, ensure_ascii=False))
    elif args.cmd == "action":
        print(json.dumps(call({"op": "act", "kind": "action", "text": args.arg}),
                         indent=2, ensure_ascii=False))
    elif args.cmd == "click":
        print(json.dumps(call({"op": "act", "kind": "click",
                               "widget": {"class": args.arg, "text": args.arg2}}),
                         indent=2, ensure_ascii=False))
    elif args.cmd == "canvas":
        r = call({"op": "canvas"})
        print(json.dumps(r, indent=2, ensure_ascii=False))
    elif args.cmd == "windows":
        print(json.dumps(call({"op": "windows"}), indent=2, ensure_ascii=False))
    elif args.cmd == "ping":
        print(json.dumps(call({"op": "ping"}), indent=2, ensure_ascii=False))
    elif args.cmd == "truth":
        print(json.dumps(call({"op": "truth"}), indent=2, ensure_ascii=False))
    elif args.cmd == "structure":
        r = call({"op": "structure"})
        print(f"widgets={len(r.get('widgets', []))} actions={len(r.get('actions', []))}")
        for w in r.get("widgets", [])[:args.limit]:
            print(f"  W {w['class']:<22} {w['text'][:28]:<28} {w['rect']}")
        for a in r.get("actions", [])[:args.limit]:
            print(f"  A {a['text'][:40]:<40} enabled={a['enabled']}")
    elif args.cmd == "act-json":
        print(json.dumps(call(json.loads(args.arg)), indent=2, ensure_ascii=False))
    elif args.cmd == "shot":
        print(json.dumps(call({"op": "shot", "path": args.arg}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
