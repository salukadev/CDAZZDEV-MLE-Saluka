# AI-ASSISTED: Grok (Cursor), Prompt: 'Implement the Task 2 and Task 3 assessment plan for a supply-chain fine-tune and a LangGraph research system', Date: 2026-10-13
"""Persistent research briefs, one file per ticker and calendar day."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

MEMORY_DIR = Path(__file__).resolve().parents[1] / "memory"


def brief_path(ticker: str, day: str | None = None) -> Path:
    stamp = day or date.today().isoformat()
    safe = "".join(ch for ch in ticker.upper() if ch.isalnum()) or "UNKNOWN"
    return MEMORY_DIR / f"{safe}_{stamp}.json"


def save_brief(ticker: str, payload: dict, day: str | None = None) -> Path:
    path = brief_path(ticker, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_brief(ticker: str, day: str | None = None) -> dict | None:
    path = brief_path(ticker, day)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
