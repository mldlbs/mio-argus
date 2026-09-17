"""
配置 A：真实大 VLM 基线适配器（P2 的核心比较）。

⚠️ 本仓库尚未运行过配置 A：当前环境没有可用的 LLM API key，也未安装
   `openai` / `anthropic` 包。此文件只提供接口，有 key 即可接入。

用法（示例）：
    set OPENAI_API_KEY=sk-...
    python vlm_adapter.py --provider openai --model gpt-4o \
        --image computer_use_data_real_balanced/balanced_click_0000.png \
        --instruction "click the save button" --mode coord

    set ANTHROPIC_API_KEY=sk-ant-...
    python vlm_adapter.py --provider anthropic --mode mark \
        --image shot.png --instruction "click save"

两种模式对应 H2 的两种动作接口：
    coord : 让模型直接输出 (x, y)         → 对应 reg（连续空间）
    mark  : 给出候选元素编号，让模型选 index → 对应 cls（离散空间）

用 standard library 发 HTTP，避免额外依赖。仅使用 `requests`（如已装）或 urllib。
"""
import argparse
import base64
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class VLMResult:
    raw: str
    action: str
    coord: Optional[Tuple[float, float]] = None   # 归一化 [0,1]
    mark: Optional[int] = None
    error: Optional[str] = None


PROMPT_COORD = (
    "You are a GUI agent. The image is a screenshot. "
    "Instruction: {instruction}\n"
    "Reply with ONLY a JSON object: "
    '{{"x": <0-1 float>, "y": <0-1 float>}} '
    "where x,y are the normalized coordinates to click."
)

PROMPT_MARK = (
    "You are a GUI agent. The image is a screenshot with numbered candidate "
    "regions (1..{n}). Instruction: {instruction}\n"
    "Reply with ONLY a JSON object: "
    '{{"mark": <integer>, "label": "<what you clicked>"}} '
    "choosing the index of the candidate that best matches the instruction."
)


# ---------- HTTP（避免额外依赖）----------

def _post_json(url: str, headers: dict, payload: dict, timeout: int = 60) -> dict:
    try:
        import requests
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except ImportError:
        pass
    import urllib.request
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ---------- 提供方 ----------

def call_openai(image_path: str, prompt: str, model: str,
                base_url: str = "https://api.openai.com/v1") -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY 未设置")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{_b64(image_path)}"}},
            ],
        }],
        "max_tokens": 256,
    }
    r = _post_json(f"{base_url}/chat/completions",
                   {"Authorization": f"Bearer {key}"}, payload)
    return r["choices"][0]["message"]["content"]


def call_anthropic(image_path: str, prompt: str, model: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY 未设置")
    payload = {
        "model": model,
        "max_tokens": 256,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": _b64(image_path)}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    r = _post_json("https://api.anthropic.com/v1/messages",
                   {"x-api-key": key, "anthropic-version": "2023-06-01"}, payload)
    return r["content"][0]["text"]


# ---------- 解析 ----------

_JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_reply(text: str, mode: str) -> VLMResult:
    m = _JSON_RE.search(text)
    if not m:
        return VLMResult(raw=text, action="unknown", error="no JSON in reply")
    try:
        d = json.loads(m.group(0))
    except Exception as e:
        return VLMResult(raw=text, action="unknown", error=f"bad JSON: {e}")

    if mode == "coord":
        if "x" in d and "y" in d:
            return VLMResult(raw=text, action="click",
                             coord=(float(d["x"]), float(d["y"])))
        return VLMResult(raw=text, action="unknown", error="missing x/y")
    if "mark" in d:
        return VLMResult(raw=text, action="click", mark=int(d["mark"]))
    return VLMResult(raw=text, action="unknown", error="missing mark")


def predict(image_path: str, instruction: str, mode: str = "coord",
            provider: str = "openai", model: Optional[str] = None,
            candidates: Optional[List] = None) -> VLMResult:
    """
    candidates: mark 模式下提供的候选元素列表（如解析器输出的 bbox）。
                仅用于提示文本；编号需与调用方对齐。
    """
    if mode == "coord":
        prompt = PROMPT_COORD.format(instruction=instruction)
    else:
        n = len(candidates) if candidates else 0
        prompt = PROMPT_MARK.format(instruction=instruction, n=n)

    if provider == "openai":
        text = call_openai(image_path, prompt, model or "gpt-4o")
    elif provider == "anthropic":
        text = call_anthropic(image_path, prompt, model or "claude-3-5-sonnet-latest")
    else:
        raise ValueError(f"未知 provider: {provider}")
    return parse_reply(text, mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--image", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--mode", default="coord", choices=["coord", "mark"])
    args = ap.parse_args()

    r = predict(args.image, args.instruction, mode=args.mode,
                provider=args.provider, model=args.model)
    print(json.dumps({
        "action": r.action, "coord": r.coord, "mark": r.mark,
        "error": r.error, "raw": r.raw[:400],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
