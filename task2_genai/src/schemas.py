# AI-ASSISTED: Grok (Cursor), Prompt: 'Implement the Task 2 and Task 3 assessment plan for a supply-chain fine-tune and a LangGraph research system', Date: 2026-10-13
"""Pydantic contracts for gold labels and the LLM judge."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnomalyExplanation(BaseModel):
    """Gold and predicted label for one supply-chain incident."""

    likely_cause: str = Field(min_length=8)
    supporting_signals: list[str] = Field(min_length=2, max_length=4)
    immediate_action: str = Field(min_length=8)
    escalation: bool
    residual_risk: str = Field(min_length=8)
    confidence: float = Field(ge=0.0, le=1.0)


class JudgeScore(BaseModel):
    """Structured score returned by the LLM-as-judge."""

    faithfulness: int = Field(ge=1, le=5)
    specificity: int = Field(ge=1, le=5)
    hallucinated: bool
    brief_reason: str = Field(min_length=8)
