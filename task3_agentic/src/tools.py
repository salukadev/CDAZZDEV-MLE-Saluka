# AI-ASSISTED: Grok (Cursor), Prompt: 'Implement the Task 2 and Task 3 assessment plan for a supply-chain fine-tune and a LangGraph research system', Date: 2026-10-13
"""The five research tools.

Each tool returns a JSON string and writes one line to agent_trace.jsonl.
Errors come back as {"ok": false, "error": ...} instead of raising, so the
agent can observe the failure and choose another tool.
"""

from __future__ import annotations

import json
import os
import time

import feedparser
import yfinance as yf
from dotenv import load_dotenv
from langchain_core.tools import tool

from task3_agentic.src.indicators import TRADING_DAYS, annualized_volatility, attach_indicators
from task3_agentic.src.prompts import SENTIMENT_SYSTEM
from task3_agentic.src.schemas import HeadlineSentiment, SentimentResult
from task3_agentic.src.tracing import log_tool_call, log_validation_failure

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
VOLATILITY_HISTORY = "2y"  # enough rows for a 90-session window, no fixed start date
MAX_SENTIMENT_HEADLINES = 12
SENTIMENT_NEUTRAL_BAND = 0.15
SEARCH_RESULTS = 5
_FORCE_EMPTY_NEWS = {"armed": False}


def arm_empty_news_once() -> None:
    """Make the next get_news call return no headlines (fallback demo)."""
    _FORCE_EMPTY_NEWS["armed"] = True


def _traced(name: str, inputs: dict, fn) -> str:
    started = time.perf_counter()
    try:
        payload = fn()
    except Exception as exc:
        payload = {"ok": False, "error": f"{name} failed: {exc}"}
    log_tool_call(name, inputs, payload, (time.perf_counter() - started) * 1000)
    return json.dumps(payload, default=str)


def _num(value) -> float | None:
    return None if value is None or value != value else float(value)  # NaN != NaN


def _price(ticker: str, period: str) -> dict:
    frame = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    frame = frame.dropna(subset=["Open", "High", "Low", "Close", "Volume"]) if not frame.empty else frame
    if frame.empty:
        return {"ok": False, "error": f"no price history for {ticker} over {period}"}
    data = attach_indicators(frame)
    last, year = data.iloc[-1], data.tail(TRADING_DAYS)
    fields = ["sma_50", "sma_200", "rsi_14", "macd", "macd_signal", "macd_histogram",
              "bollinger_upper", "bollinger_middle", "bollinger_lower"]
    return {
        "ok": True,
        "ticker": ticker.upper(),
        "as_of": str(data.index[-1].date()),
        "period": period,
        "rows": len(data),
        "current_price": float(last["Close"]),
        "high_52w": _num(year["High"].max()),
        "low_52w": _num(year["Low"].min()),
        **{field: _num(last[field]) for field in fields},
    }


def _news(ticker: str, n: int) -> dict:
    if _FORCE_EMPTY_NEWS["armed"]:
        _FORCE_EMPTY_NEWS["armed"] = False
        return {"ok": True, "ticker": ticker.upper(), "source": "forced_empty", "count": 0, "headlines": []}
    titles: list[str] = []
    for item in yf.Ticker(ticker).news or []:
        content = item.get("content", item) if isinstance(item, dict) else {}
        if content.get("title"):
            titles.append(content["title"].strip())
    if len(titles) < n:  # top up from the public Yahoo Finance RSS feed
        feed = feedparser.parse(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US")
        titles += [entry.title.strip() for entry in feed.entries if entry.get("title")]
    titles = list(dict.fromkeys(titles))[:n]
    if not titles:
        return {"ok": False, "error": f"no headlines for {ticker}", "headlines": []}
    return {"ok": True, "ticker": ticker.upper(), "count": len(titles), "headlines": titles}


def _volatility(ticker: str, window: int) -> dict:
    close = yf.Ticker(ticker).history(period=VOLATILITY_HISTORY, auto_adjust=False)["Close"].dropna()
    value = annualized_volatility(close, window)
    if value is None:
        return {"ok": False, "error": f"not enough data for window={window}"}
    return {"ok": True, "ticker": ticker.upper(), "window": window, "annualized_volatility": value, "observations": window}


def _sentiment(headlines: list[str]) -> dict:
    cleaned = [h.strip() for h in headlines if isinstance(h, str) and h.strip()][:MAX_SENTIMENT_HEADLINES]
    if not cleaned:
        return {"ok": False, "error": "no headlines to score"}
    if not os.environ.get("GROQ_API_KEY"):
        return {"ok": False, "error": "GROQ_API_KEY is not set"}
    from groq import Groq

    error = ""
    for attempt in (1, 2):
        try:
            response = Groq().chat.completions.create(
                model=GROQ_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SENTIMENT_SYSTEM},
                    {"role": "user", "content": json.dumps({"headlines": cleaned})},
                ],
            )
            items = [HeadlineSentiment.model_validate(i) for i in json.loads(response.choices[0].message.content)["items"]]
            if not items:
                raise ValueError("no items returned")
            # Confidence-weighted mean of +1 / 0 / -1.
            sign = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            score = sum(sign[i.sentiment] * i.confidence for i in items) / max(sum(i.confidence for i in items), 1e-9)
            label = "positive" if score > SENTIMENT_NEUTRAL_BAND else "negative" if score < -SENTIMENT_NEUTRAL_BAND else "neutral"
            result = SentimentResult(items=items, overall_sentiment=label, overall_score=round(score, 4))
            return {"ok": True, **result.model_dump()}
        except Exception as exc:
            error = str(exc)
            log_validation_failure(f"llm_sentiment attempt {attempt}: {error}")
    return {"ok": False, "error": f"sentiment validation failed: {error}"}


def _search(query: str) -> dict:
    try:
        from ddgs import DDGS  # current name of the duckduckgo-search package
    except ImportError:
        from duckduckgo_search import DDGS
    hits = DDGS().text(query, max_results=SEARCH_RESULTS) or []
    results = [{"title": h.get("title", ""), "snippet": h.get("body", ""), "link": h.get("href", "")} for h in hits]
    if not results:
        return {"ok": False, "error": "web search returned no results", "results": []}
    return {"ok": True, "query": query, "results": results}


@tool
def get_price_data(ticker: str, period: str = "2y") -> str:
    """Daily OHLCV plus SMA50, SMA200, RSI14, MACD(12,26,9) and Bollinger(20,2). period is a yfinance window like 1y or 2y."""
    return _traced("get_price_data", {"ticker": ticker, "period": period}, lambda: _price(ticker, period))


@tool
def get_news(ticker: str, n: int = 10) -> str:
    """Recent headlines for a ticker. If the list is empty, use web_search instead of stopping."""
    return _traced("get_news", {"ticker": ticker, "n": n}, lambda: _news(ticker, n))


@tool
def calculate_volatility(ticker: str, window: int = 30) -> str:
    """Annualised historical volatility from daily log returns over `window` trading sessions."""
    return _traced("calculate_volatility", {"ticker": ticker, "window": window}, lambda: _volatility(ticker, window))


@tool
def llm_sentiment(headlines: list[str]) -> str:
    """Per-headline sentiment (label, confidence, reason) and an overall score in [-1, 1]."""
    return _traced("llm_sentiment", {"headlines": headlines}, lambda: _sentiment(headlines))


@tool
def web_search(query: str) -> str:
    """Search the web for analyst commentary. Use it when headlines are missing or a risk needs a source."""
    return _traced("web_search", {"query": query}, lambda: _search(query))


SINGLE_AGENT_TOOLS = [get_price_data, get_news, calculate_volatility, llm_sentiment, web_search]
ANALYST_TOOLS = [get_price_data, calculate_volatility, llm_sentiment]
WRITER_TOOLS = [web_search, get_news]


def call_tool(tool_obj, **kwargs) -> dict:
    return json.loads(tool_obj.invoke(kwargs))
