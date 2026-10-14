# AI-ASSISTED: Grok (Cursor), Prompt: 'Implement the Task 2 and Task 3 assessment plan for a supply-chain fine-tune and a LangGraph research system', Date: 2026-10-13
"""CLI and printing helpers shared with the notebook.

    python -m task3_agentic.src.pipeline --ticker AAPL --mode multi
"""

from __future__ import annotations

import argparse

from task3_agentic.src.graph import run_multi_agent, run_single_agent


def print_messages(messages) -> None:
    """Print the ReAct loop: each tool call, its observation, and the final answer."""
    for m in messages:
        kind = m.__class__.__name__
        for call in getattr(m, "tool_calls", None) or []:
            print(f"[{kind}] TOOL CALL {call['name']}({call['args']})")
        if m.content:
            text = m.content if isinstance(m.content, str) else str(m.content)
            print(f"[{kind}] {text if kind == 'AIMessage' else text[:300]}\n")


def print_report(report: dict) -> None:
    print("## Financial Health Summary\n" + report["financial_health_summary"] + "\n")
    print("## Top Three Risks")
    for i, risk in enumerate(report["top_three_risks"], 1):
        print(f"{i}. {risk['name']} ({risk['severity']}): {risk['evidence']}")
    print("\n## Hedge Strategy Recommendation\n" + report["hedge_strategy"])
    print("\nClarification used: " + report["used_clarification"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--mode", choices=["multi", "single"], default="multi")
    parser.add_argument("--ignore-cache", action="store_true")
    args = parser.parse_args()
    if args.mode == "single":
        print_messages(run_single_agent(args.ticker)[1])
        return
    result = run_multi_agent(args.ticker, use_cache=not args.ignore_cache)
    print(f"cache_hit={result['cache_hit']}")
    print("\n".join(result.get("steps", [])))
    if result.get("final_report"):
        print(f"\nreport_source={result.get('report_source')}\n")
        print_report(result["final_report"])


if __name__ == "__main__":
    main()
