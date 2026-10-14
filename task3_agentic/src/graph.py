# AI-ASSISTED: Grok (Cursor), Prompt: 'Implement the Task 2 and Task 3 assessment plan for a supply-chain fine-tune and a LangGraph research system', Date: 2026-10-13
"""3A: single ReAct research agent. 3B: two-agent LangGraph with a critique loop.

Agent A (data analyst) may only use ANALYST_TOOLS; Agent B (research writer)
may only use WRITER_TOOLS. Every tool call goes through `_use`, which rejects
tools outside the calling agent's list. A hands B a Pydantic DataBrief.
"""

from __future__ import annotations

import json
from typing import TypedDict

from dotenv import load_dotenv

from task3_agentic.src.memory import load_brief, save_brief
from task3_agentic.src.prompts import ANALYST_NOTES_SYSTEM, CLARIFICATION_QUESTION, SINGLE_AGENT_SYSTEM, WRITER_SYSTEM
from task3_agentic.src.schemas import (
    ClarificationRequest,
    ClarificationResponse,
    DataBrief,
    PriceSnapshot,
    ResearchReport,
    RiskFlag,
    SentimentResult,
    VolatilityResult,
)
from task3_agentic.src.tools import (
    ANALYST_TOOLS,
    GROQ_MODEL,
    SINGLE_AGENT_TOOLS,
    WRITER_TOOLS,
    arm_empty_news_once,
    calculate_volatility,
    call_tool,
    get_news,
    get_price_data,
    llm_sentiment,
    web_search,
)
from task3_agentic.src.tracing import set_agent

load_dotenv()

ANALYST_WINDOW = 30  # default lookback; deliberately shorter than the 90-day horizon
HORIZON_WINDOW = 90
HIGH_VOL, MEDIUM_VOL = 0.40, 0.25
RSI_OVERBOUGHT, RSI_OVERSOLD = 70, 30
DEEP_DRAWDOWN = 0.20
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
QUERY = (
    "Analyse the current financial health and market sentiment of {ticker}. Identify the top three "
    "risks to its share price over the next 90 days and suggest one data-driven hedge strategy."
)


def _use(allowed: list, tool_obj, **kwargs) -> dict:
    if tool_obj not in allowed:
        raise PermissionError(f"{tool_obj.name} is not in this agent's tool list")
    return call_tool(tool_obj, **kwargs)


def _groq_json(system: str, payload: dict) -> dict:
    import time
    from groq import Groq

    last_error = "no attempt"
    for attempt in range(1, 5):
        try:
            response = Groq().chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, default=str)}],
            )
            return json.loads(response.choices[0].message.content)
        except Exception as exc:
            last_error = str(exc)
            if "429" not in last_error and "rate" not in last_error.lower():
                raise
            time.sleep(15 * attempt)
    raise RuntimeError(last_error)


def quantitative_risks(price: dict, vol: dict, sentiment: SentimentResult) -> list[RiskFlag]:
    """Rule-based risk flags from the indicators, most severe first, top three."""
    spot, flags = price["current_price"], []
    if price.get("sma_200") and spot < price["sma_200"]:
        flags.append(RiskFlag(name="Below 200-day average", severity="high",
                              evidence=f"Price {spot:.2f} < SMA200 {price['sma_200']:.2f}."))
    rsi = price.get("rsi_14")
    if rsi is not None and (rsi >= RSI_OVERBOUGHT or rsi <= RSI_OVERSOLD):
        flags.append(RiskFlag(name="Stretched momentum", severity="medium", evidence=f"RSI(14) is {rsi:.1f}."))
    if price.get("macd_histogram") is not None and price["macd_histogram"] < 0:
        flags.append(RiskFlag(name="MACD below signal", severity="medium",
                              evidence=f"MACD histogram is {price['macd_histogram']:.3f}."))
    if price.get("bollinger_upper") and spot > price["bollinger_upper"]:
        flags.append(RiskFlag(name="Above upper Bollinger band", severity="medium",
                              evidence=f"Price {spot:.2f} > upper band {price['bollinger_upper']:.2f}."))
    annual = vol["annualized_volatility"]
    flags.append(RiskFlag(
        name="Volatility regime",
        severity="high" if annual >= HIGH_VOL else "medium" if annual >= MEDIUM_VOL else "low",
        evidence=f"{vol['window']}-session annualised volatility is {annual:.1%}.",
    ))
    if sentiment.overall_sentiment == "negative":
        flags.append(RiskFlag(name="Negative headlines", severity="medium",
                              evidence=f"Aggregate sentiment score {sentiment.overall_score}."))
    if price.get("high_52w"):
        drawdown = 1 - spot / price["high_52w"]
        flags.append(RiskFlag(name="Distance from 52-week high", severity="high" if drawdown >= DEEP_DRAWDOWN else "low",
                              evidence=f"Price is {drawdown:.1%} below the 52-week high {price['high_52w']:.2f}."))
    return sorted(flags, key=lambda f: SEVERITY_ORDER[f.severity])[:3]


# ---------- 3B: two-agent graph ----------

class ResearchState(TypedDict, total=False):
    ticker: str
    headlines: list[str]
    search_results: list[dict]
    data_brief: dict
    clarification_request: dict
    clarification_response: dict
    final_report: dict
    report_source: str
    steps: list[str]
    error: str


def writer_gather(state: ResearchState) -> dict:
    """Agent B collects headlines; if none, it falls back to web search."""
    set_agent("research_writer")
    news = _use(WRITER_TOOLS, get_news, ticker=state["ticker"], n=10)
    steps = [f"[writer] get_news -> ok={news['ok']} count={news.get('count', 0)}"]
    headlines, results = news.get("headlines", []), []
    if not headlines:
        search = _use(WRITER_TOOLS, web_search, query=f"{state['ticker']} stock news")
        results = search.get("results", [])
        headlines = [r["title"] for r in results if r["title"]]
        steps.append(f"[writer] no headlines, fell back to web_search -> {len(headlines)} titles")
    return {"headlines": headlines, "search_results": results, "steps": state["steps"] + steps}


def analyst_brief(state: ResearchState) -> dict:
    """Agent A builds the quantitative DataBrief."""
    set_agent("data_analyst")
    ticker, steps = state["ticker"], list(state["steps"])
    price = _use(ANALYST_TOOLS, get_price_data, ticker=ticker, period="2y")
    if not price["ok"]:
        price = _use(ANALYST_TOOLS, get_price_data, ticker=ticker, period="1y")
        steps.append("[analyst] 2y history failed, retried with 1y")
    vol = _use(ANALYST_TOOLS, calculate_volatility, ticker=ticker, window=ANALYST_WINDOW)
    if not (price["ok"] and vol["ok"]):
        return {"error": price.get("error") or vol.get("error"), "steps": steps + ["[analyst] price data unavailable, stopping"]}
    sent = _use(ANALYST_TOOLS, llm_sentiment, headlines=state["headlines"])
    sentiment = SentimentResult.model_validate(sent) if sent["ok"] else SentimentResult(items=[], overall_sentiment="neutral", overall_score=0.0)
    steps.append(f"[analyst] price ok, vol({ANALYST_WINDOW})={vol['annualized_volatility']:.2%}, "
                 f"sentiment={sentiment.overall_sentiment if sent['ok'] else 'unavailable: ' + sent['error']}")
    numbers = {k: price.get(k) for k in ("current_price", "sma_50", "sma_200", "rsi_14", "macd_histogram", "bollinger_upper", "bollinger_lower")}
    try:
        notes = _groq_json(ANALYST_NOTES_SYSTEM, {**numbers, "volatility": vol, "sentiment": sentiment.overall_sentiment})["notes"]
    except Exception as exc:
        notes = "Indicator note unavailable (LLM call failed)."
        steps.append(f"[analyst] notes LLM call failed: {str(exc)[:120]}")
    brief = DataBrief(
        ticker=ticker, as_of=price["as_of"], price=PriceSnapshot.model_validate(price),
        volatility=VolatilityResult.model_validate(vol), sentiment=sentiment,
        quantitative_risks=quantitative_risks(price, vol, sentiment), notes=notes,
    )
    steps.append("[analyst -> writer] DataBrief handed off:\n" + brief.model_dump_json(indent=2, exclude={"sentiment": {"items"}}))
    return {"data_brief": brief.model_dump(), "steps": steps}


def writer_review(state: ResearchState) -> dict:
    """Agent B critiques the brief: a 30-session window does not cover a 90-day horizon."""
    set_agent("research_writer")
    brief = DataBrief.model_validate(state["data_brief"])
    if brief.volatility.window >= HORIZON_WINDOW:
        return {"steps": state["steps"] + ["[writer] brief accepted without clarification"]}
    request = ClarificationRequest(missing_field="annualized_volatility_90d",
                                   question=CLARIFICATION_QUESTION.format(window=brief.volatility.window))
    return {"clarification_request": request.model_dump(),
            "steps": state["steps"] + [f"[writer -> analyst] clarification request: {request.model_dump_json()}"]}


def analyst_clarify(state: ResearchState) -> dict:
    """Agent A answers the clarification with a new tool call."""
    set_agent("data_analyst")
    brief = DataBrief.model_validate(state["data_brief"])
    vol = _use(ANALYST_TOOLS, calculate_volatility, ticker=state["ticker"], window=HORIZON_WINDOW)
    if vol["ok"]:
        old, new = brief.volatility.annualized_volatility, vol["annualized_volatility"]
        answer = f"{HORIZON_WINDOW}-session volatility is {new:.2%}, {'higher' if new > old else 'lower'} than the {brief.volatility.window}-session {old:.2%}."
        result = VolatilityResult.model_validate(vol)
    else:
        answer, result = f"90-session volatility unavailable ({vol['error']}); keep the {brief.volatility.window}-session figure.", brief.volatility
    response = ClarificationResponse(missing_field="annualized_volatility_90d", answer=answer, volatility=result)
    return {"clarification_response": response.model_dump(),
            "steps": state["steps"] + [f"[analyst -> writer] clarification response: {response.model_dump_json()}"]}


def _fallback_report(brief: DataBrief, clar: ClarificationResponse | None) -> ResearchReport:
    vol = clar.volatility if clar else brief.volatility
    spot, p = brief.price.current_price, brief.price
    one_sigma = spot * vol.annualized_volatility * (HORIZON_WINDOW / 252) ** 0.5
    fmt = lambda x: "n/a" if x is None else f"{x:.2f}"  # noqa: E731
    return ResearchReport(
        ticker=brief.ticker,
        financial_health_summary=(f"{brief.ticker} trades at {spot:.2f} (SMA50 {fmt(p.sma_50)}, SMA200 {fmt(p.sma_200)}, "
                                  f"RSI {fmt(p.rsi_14)}). {brief.notes}"),
        top_three_risks=brief.quantitative_risks,
        hedge_strategy=(f"Buy a ~90-day protective put struck about one sigma below spot: with {vol.window}-session "
                        f"volatility of {vol.annualized_volatility:.1%}, one sigma over 90 days is ~{one_sigma:.2f}, "
                        f"so strike near {spot - one_sigma:.2f}."),
        used_clarification=clar.answer if clar else "No clarification was needed.",
    )


def writer_report(state: ResearchState) -> dict:
    """Agent B writes the final report from the brief, the clarification, headlines, and web search."""
    set_agent("research_writer")
    brief = DataBrief.model_validate(state["data_brief"])
    clar = ClarificationResponse.model_validate(state["clarification_response"]) if state.get("clarification_response") else None
    search = _use(WRITER_TOOLS, web_search, query=f"{state['ticker']} stock risks analyst")
    steps = state["steps"] + [f"[writer] web_search -> ok={search['ok']} results={len(search.get('results', []))}"]
    try:
        payload = _groq_json(WRITER_SYSTEM, {"data_brief": state["data_brief"], "clarification": state.get("clarification_response"),
                                             "headlines": state["headlines"], "web_search": search.get("results", [])})
        report, source = ResearchReport.model_validate({**payload, "ticker": brief.ticker}), "llm"
    except Exception as exc:
        steps.append(f"[writer] LLM report unavailable or failed validation ({str(exc)[:120]}); using rule-based report")
        report, source = _fallback_report(brief, clar), "fallback"
    return {"final_report": report.model_dump(), "report_source": source, "steps": steps}


def build_multi_agent_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(ResearchState)
    for name, node in [("writer_gather", writer_gather), ("analyst_brief", analyst_brief), ("writer_review", writer_review),
                       ("analyst_clarify", analyst_clarify), ("writer_report", writer_report)]:
        graph.add_node(name, node)
    graph.add_edge(START, "writer_gather")
    graph.add_edge("writer_gather", "analyst_brief")
    graph.add_conditional_edges("analyst_brief", lambda s: END if s.get("error") else "writer_review")
    graph.add_conditional_edges("writer_review", lambda s: "analyst_clarify" if s.get("clarification_request") else "writer_report")
    graph.add_edge("analyst_clarify", "writer_report")
    graph.add_edge("writer_report", END)
    return graph.compile()


def run_multi_agent(ticker: str, use_cache: bool = True) -> dict:
    """Run the two-agent pipeline, or load today's cached brief for this ticker."""
    ticker = ticker.upper()
    if use_cache and (cached := load_brief(ticker)):
        return {**cached, "cache_hit": True}
    state = build_multi_agent_graph().invoke({"ticker": ticker, "steps": []})
    if state.get("final_report"):
        save_brief(ticker, state)
    return {**state, "cache_hit": False}


# ---------- 3A: single ReAct agent with short-term memory ----------

def build_single_agent():
    from langchain_groq import ChatGroq
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.prebuilt import create_react_agent

    # The checkpointer keeps the message history (including tool results) per thread_id.
    return create_react_agent(ChatGroq(model=GROQ_MODEL, temperature=0), SINGLE_AGENT_TOOLS,
                              prompt=SINGLE_AGENT_SYSTEM, checkpointer=InMemorySaver())


def ask(agent, question: str, thread_id: str) -> list:
    set_agent("single_agent")
    result = agent.invoke({"messages": [("user", question)]},
                          config={"configurable": {"thread_id": thread_id}, "recursion_limit": 25})
    return result["messages"]


def run_single_agent(ticker: str, demo_empty_news: bool = False):
    if demo_empty_news:
        arm_empty_news_once()
    agent = build_single_agent()
    return agent, ask(agent, QUERY.format(ticker=ticker.upper()), thread_id=f"{ticker}-session")
