# AI-ASSISTED: Grok (Cursor), Prompt: 'Implement the Task 2 and Task 3 assessment plan for a supply-chain fine-tune and a LangGraph research system', Date: 2026-10-13
"""agent_trace.jsonl logging plus the in-session (short-term) memory."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
TRACE_PATH = LOG_DIR / "agent_trace.jsonl"
VALIDATION_LOG = LOG_DIR / "validation.log"
OUTPUT_PREVIEW_CHARS = 200

# Set by each graph node before it calls a tool, so the trace names the agent.
CURRENT_AGENT = {"name": "unspecified"}
# Last successful result of each tool in this session.
SESSION_MEMORY: dict[str, dict] = {}


def set_agent(name: str) -> None:
    CURRENT_AGENT["name"] = name


def log_tool_call(tool: str, inputs: dict, output: dict, duration_ms: float) -> None:
    if output.get("ok"):
        SESSION_MEMORY[tool] = output
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": CURRENT_AGENT["name"],
        "tool": tool,
        "inputs": inputs,
        "output": json.dumps(output, default=str)[:OUTPUT_PREVIEW_CHARS],
        "duration_ms": round(duration_ms, 2),
    }
    with TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def log_validation_failure(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with VALIDATION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def count_trace_lines() -> int:
    if not TRACE_PATH.exists():
        return 0
    return sum(1 for line in TRACE_PATH.read_text(encoding="utf-8").splitlines() if line.strip())
