"""episode schema：原始观测记录（不含任何原因/诊断判断）。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass
class Frame:
    """一次采样：画面文件 + 文本 + 取证通道。"""
    frame: str | None = None          # 相对 results 目录的 png 路径
    text: str = ""
    ocr_channel: str = "full"         # full | fast | none
    ts: float = 0.0


@dataclass
class Episode:
    id: str
    # 采集程序（不是结论，归因不使用）
    origin: str
    action: dict = field(default_factory=dict)     # {type, x, y, rect, text_target}
    before: Frame = field(default_factory=Frame)
    after: Frame = field(default_factory=Frame)
    expected: dict = field(default_factory=dict)   # {kind: selection_change|text_change|ocr_text}
    extra: dict = field(default_factory=dict)
    schema: int = SCHEMA_VERSION

    def to_json(self) -> dict:
        d = asdict(self)
        d["schema"] = self.schema
        return d


def write_episodes(path: Path, episodes: list[Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep.to_json(), ensure_ascii=False) + "\n")


def read_episodes(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
