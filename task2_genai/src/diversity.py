# AI-ASSISTED: Grok (Cursor), Prompt: 'Implement the Task 2 and Task 3 assessment plan for a supply-chain fine-tune and a LangGraph research system', Date: 2026-10-13
"""Diversity statistics for the generated training prompts.

The assessment rejects a set in which most rows are small edits of one
scenario. These helpers measure length, topic mix, and near-duplicates.
"""

from __future__ import annotations

import re
from collections import Counter


_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def jaccard(left: set[str], right: set[str]) -> float:
    union = len(left | right)
    if union == 0:
        return 0.0
    return len(left & right) / union


def length_stats(texts: list[str]) -> dict:
    lengths = sorted(len(text) for text in texts)
    if not lengths:
        return {"n": 0, "min": 0, "median": 0, "p90": 0, "max": 0}

    def percentile(p: float) -> int:
        index = round((p / 100) * (len(lengths) - 1))
        return lengths[max(0, min(len(lengths) - 1, index))]

    return {
        "n": len(lengths),
        "min": lengths[0],
        "median": percentile(50),
        "p90": percentile(90),
        "max": lengths[-1],
    }


def frequency(values: list[str]) -> dict[str, int]:
    return dict(Counter(values).most_common())


def max_share(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    return max(counts.values()) / len(values)


def near_duplicate_rate(texts: list[str], threshold: float = 0.8) -> dict:
    """Share of prompts whose nearest neighbour exceeds the Jaccard threshold."""
    parsed = [tokens(text) for text in texts]
    nearest: list[float] = []
    for index, current in enumerate(parsed):
        best = 0.0
        for other_index, other in enumerate(parsed):
            if index == other_index:
                continue
            best = max(best, jaccard(current, other))
        nearest.append(best)
    flagged = sum(score >= threshold for score in nearest)
    rate = flagged / len(texts) if texts else 0.0
    return {
        "threshold": threshold,
        "near_duplicate_rate": round(rate, 4),
        "flagged": flagged,
        "max_nearest_jaccard": round(max(nearest), 4) if nearest else 0.0,
        "median_nearest_jaccard": round(sorted(nearest)[len(nearest) // 2], 4) if nearest else 0.0,
    }


def diversity_report(records: list[dict]) -> dict:
    """records need user_text, event_type, and industry."""
    texts = [row["user_text"] for row in records]
    event_types = [row["event_type"] for row in records]
    industries = [row["industry"] for row in records]
    return {
        "count": len(records),
        "prompt_length_chars": length_stats(texts),
        "event_type_counts": frequency(event_types),
        "event_type_max_share": round(max_share(event_types), 4),
        "industry_counts": frequency(industries),
        "industry_max_share": round(max_share(industries), 4),
        "near_duplicates": near_duplicate_rate(texts),
    }
