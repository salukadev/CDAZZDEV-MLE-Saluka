# AI-ASSISTED: Grok (Cursor), Prompt: 'Implement the Task 2 and Task 3 assessment plan for a supply-chain fine-tune and a LangGraph research system', Date: 2026-10-13
"""Label the scenario grid with a Groq teacher and write Qwen chat JSONL.

Teacher: Llama 3.3 70B on Groq. Student: Qwen2.5 (a different model family).
The JSONL system turn is the student prompt, the same one used at evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from task2_genai.src.diversity import diversity_report
from task2_genai.src.scenarios import build_scenarios
from task2_genai.src.schemas import AnomalyExplanation

ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT / "task2_genai" / "prompts"
DATA_DIR = ROOT / "task2_genai" / "data"
MAX_TYPE_SHARE = 0.20
MAX_NEAR_DUPLICATE_RATE = 0.5
TEACHER_ATTEMPTS = 3

logger = logging.getLogger("task2.generate")


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("response did not contain a JSON object")
    return json.loads(text[start : end + 1])


def label_one(client, model: str, teacher_prompt: str, user_text: str) -> AnomalyExplanation:
    """Call the teacher and validate with Pydantic; retry on any failure."""
    error = ""
    for attempt in range(1, TEACHER_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": teacher_prompt},
                    {"role": "user", "content": user_text},
                ],
            )
            return AnomalyExplanation.model_validate(extract_json(response.choices[0].message.content or ""))
        except Exception as exc:
            error = str(exc)
            logger.warning("teacher attempt %s failed: %s", attempt, error)
            time.sleep(2**attempt)
    raise RuntimeError(f"teacher failed after {TEACHER_ATTEMPTS} attempts: {error}")


def stratified_split(rows: list[dict], seed: int = 7) -> dict[str, list[dict]]:
    """80/10/10 within each event type (8/1/1 for 10 rows per type)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["event_type"]].append(row)
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for event_type in sorted(groups):
        bucket = groups[event_type]
        rng.shuffle(bucket)
        n_hold = max(1, round(len(bucket) * 0.10))
        splits["test"] += bucket[:n_hold]
        splits["val"] += bucket[n_hold : 2 * n_hold]
        splits["train"] += bucket[2 * n_hold :]
    return splits


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate(seed: int = 7, dry_run: bool = False, limit: int | None = None) -> dict:
    load_dotenv(ROOT / ".env")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scenarios = build_scenarios(seed=seed)[:limit]
    report = diversity_report(scenarios)
    (DATA_DIR / "diversity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["event_type_max_share"] > MAX_TYPE_SHARE:
        raise RuntimeError("one event type exceeds 20 percent of the set")
    if report["near_duplicates"]["near_duplicate_rate"] > MAX_NEAR_DUPLICATE_RATE:
        raise RuntimeError("most prompts are near-duplicates")
    if dry_run:
        print(json.dumps(report, indent=2))
        return report

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env and rerun.")
    from groq import Groq

    client = Groq()
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    teacher_prompt, student_prompt = load_prompt("teacher_system.txt"), load_prompt("student_system.txt")
    labelled = []
    for index, row in enumerate(scenarios, start=1):
        label = label_one(client, model, teacher_prompt, row["user_text"])
        labelled.append({
            **row,
            "messages": [
                {"role": "system", "content": student_prompt},
                {"role": "user", "content": row["user_text"]},
                {"role": "assistant", "content": json.dumps(label.model_dump(), ensure_ascii=False)},
            ],
        })
        logger.info("labelled %s/%s %s", index, len(scenarios), row["example_id"])

    splits = stratified_split(labelled, seed=seed)
    for name, rows in splits.items():
        write_jsonl(DATA_DIR / f"{name}.jsonl", [{"messages": r["messages"]} for r in rows])
        write_jsonl(
            DATA_DIR / f"{name}_meta.jsonl",
            [{k: r[k] for k in ("example_id", "event_type", "industry")} for r in rows],
        )
    summary = {
        "teacher_model": model,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "diversity": report,
    }
    (DATA_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the supply-chain fine-tune dataset")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true", help="report diversity without calling Groq")
    parser.add_argument("--limit", type=int, default=None, help="label only the first N scenarios")
    args = parser.parse_args()
    generate(seed=args.seed, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
