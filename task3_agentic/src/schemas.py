# AI-ASSISTED: Grok (Cursor), Prompt: 'Implement the Task 2 and Task 3 assessment plan for a supply-chain fine-tune and a LangGraph research system', Date: 2026-10-13
"""Structured contracts passed between agents. Handoffs are not raw strings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HeadlineSentiment(BaseModel):
    headline: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    brief_reason: str


class SentimentResult(BaseModel):
    items: list[HeadlineSentiment]
    overall_sentiment: Literal["positive", "negative", "neutral"]
    overall_score: float = Field(ge=-1.0, le=1.0)


class PriceSnapshot(BaseModel):
    ticker: str
    as_of: str
    period: str
    rows: int
    current_price: float
    high_52w: float | None = None
    low_52w: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bollinger_upper: float | None = None
    bollinger_middle: float | None = None
    bollinger_lower: float | None = None


class VolatilityResult(BaseModel):
    ticker: str
    window: int
    annualized_volatility: float
    observations: int


class RiskFlag(BaseModel):
    name: str
    evidence: str
    severity: Literal["low", "medium", "high"]


class DataBrief(BaseModel):
    """Agent A output. Agent B must consume this object, not a free-form blob."""

    ticker: str
    as_of: str
    price: PriceSnapshot
    volatility: VolatilityResult
    sentiment: SentimentResult
    quantitative_risks: list[RiskFlag]
    notes: str


class ClarificationRequest(BaseModel):
    missing_field: str
    question: str


class ClarificationResponse(BaseModel):
    missing_field: str
    answer: str
    volatility: VolatilityResult


class ResearchReport(BaseModel):
    ticker: str
    financial_health_summary: str
    top_three_risks: list[RiskFlag] = Field(min_length=3, max_length=3)
    hedge_strategy: str
    used_clarification: str
