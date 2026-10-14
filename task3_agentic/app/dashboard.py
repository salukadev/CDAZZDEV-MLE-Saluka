# AI-ASSISTED: Grok (Cursor), Prompt: 'Implement the Task 2 and Task 3 assessment plan for a supply-chain fine-tune and a LangGraph research system', Date: 2026-10-13
"""Streamlit view of task3_agentic/logs/agent_trace.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

TRACE_PATH = Path(__file__).resolve().parents[1] / "logs" / "agent_trace.jsonl"


def load_rows() -> list[dict]:
    if not TRACE_PATH.exists():
        return []
    rows = []
    for line in TRACE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    st.set_page_config(page_title="Agent trace", layout="wide")
    st.title("Agent trace")
    st.caption(str(TRACE_PATH))
    rows = load_rows()
    if not rows:
        st.info("No trace yet. Run the research agent to append tool calls.")
        return
    frame = pd.DataFrame(rows)
    st.metric("Tool calls", len(frame))
    if "duration_ms" in frame.columns:
        summary = frame.groupby("tool", as_index=False)["duration_ms"].sum()
        st.bar_chart(summary, x="tool", y="duration_ms")
    st.dataframe(frame, use_container_width=True)


if __name__ == "__main__":
    main()
