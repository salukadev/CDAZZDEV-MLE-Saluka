# AI-ASSISTED: Grok (Cursor), Prompt: 'Implement the Task 2 and Task 3 assessment plan for a supply-chain fine-tune and a LangGraph research system', Date: 2026-10-13
"""ROUGE-L, LLM-as-judge, and hallucination-rate helpers used by the notebook."""

from __future__ import annotations

import logging

from rouge_score import rouge_scorer

from task2_genai.src.generate_dataset import extract_json, load_prompt
from task2_genai.src.schemas import JudgeScore

logger = logging.getLogger("task2.evaluate")
_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
REVIEW_LABELS = {"correct", "partially_correct", "hallucinated"}
MIN_REVIEWS = 10


def mean_rouge_l(predictions: list[str], references: list[str]) -> float:
    scores = [_SCORER.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    return sum(scores) / len(scores) if scores else 0.0


def judge_one(client, model: str, incident: str, gold: str, prediction: str) -> JudgeScore | None:
    """Score one answer. Validation failures are logged, retried once, then return None."""
    user = f"INCIDENT:\n{incident}\n\nGOLD:\n{gold}\n\nMODEL:\n{prediction}"
    for attempt in (1, 2):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": load_prompt("judge_system.txt")},
                    {"role": "user", "content": user},
                ],
            )
            return JudgeScore.model_validate(extract_json(response.choices[0].message.content or ""))
        except Exception as exc:
            logger.warning("judge attempt %s failed: %s", attempt, exc)
    return None


def hallucination_rate(labels: list[str]) -> float:
    if len(labels) < MIN_REVIEWS or not set(labels) <= REVIEW_LABELS:
        raise ValueError(f"need at least {MIN_REVIEWS} labels from {sorted(REVIEW_LABELS)}")
    return labels.count("hallucinated") / len(labels)


def parse_confidence(text: str) -> float | None:
    """Self-rated confidence from a JSON answer, or None if it cannot be read."""
    try:
        value = float(extract_json(text)["confidence"])
    except Exception:
        return None
    return value if 0.0 <= value <= 1.0 else None
