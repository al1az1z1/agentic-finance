from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import pandas as pd

from ..config.settings import SETTINGS
from ..data_io.prices import fetch_prices
from ..data_io.news import fetch_news
from ..analysis.signals import fetch_indicator
from ..analysis.text import preprocess_news, add_tags_and_numbers, recent_topk, summarize_rows
from ..system.router import choose_agents
from ..system.memory import append_memory
from ..agents import NewsAnalysisAgent, MarketSignalsAgent, RiskAssessmentAgent, SynthesisAgent, CritiqueAgent, AgentResponse

@dataclass
class OrchestratorResult:
    plan: List[str]
    evidence: Dict[str, Any]
    agent_outputs: List[AgentResponse]
    final: AgentResponse
    critique: AgentResponse

def run_pipeline(symbol: str, start: str | None, end: str | None, required_tags: list[str] | None = None) -> OrchestratorResult:
    plan = [
        "fetch_prices", "fetch_news", "preprocess", "classify_extract",
        "retrieve_topk", "route", "run_agents", "synthesize", "critique", "save_memory"
    ]

    # 1) fetch
    prices = fetch_prices(symbol, start, end)
    news = fetch_news(symbol)

    # 2) preprocess chain
    news_pp = add_tags_and_numbers(preprocess_news(news))

    # 3) retrieval (deterministic)
    top_news = recent_topk(news_pp, SETTINGS.topk_news, SETTINGS.news_window_days, required_tags)

    # 4) route
    has_news = not top_news.empty
    has_prices = not prices.empty
    rsi = fetch_indicator(symbol, "RSI", 14)
    sma = fetch_indicator(symbol, "SMA", 20)
    has_technicals = (not rsi.empty) or (not sma.empty)
    lanes = choose_agents(has_news, has_prices, has_technicals)

    # 5) run agents
    outputs: List[AgentResponse] = []
    if "news" in lanes:
        bullets = summarize_rows(top_news)
        outputs.append(NewsAnalysisAgent().process(symbol, bullets))
    if "technical" in lanes:
        technicals = {
            "rsi": (rsi["RSI"].iloc[-1] if not rsi.empty else None),
            "sma20": (sma["SMA"].iloc[-1] if not sma.empty else None),
            "current_price": (prices["close"].iloc[-1] if not prices.empty else None),
            "volume": (int(prices["volume"].iloc[-1]) if not prices.empty else None),
        }
        outputs.append(MarketSignalsAgent().process(symbol, technicals))
    # always run risk (simple proxy)
    risk = {
        "beta": None, "volatility_30d": float(prices["close"].pct_change().tail(30).std()*100) if not prices.empty else None
    }
    outputs.append(RiskAssessmentAgent().process(symbol, risk))

    # 6) synthesize + critique
    synth = SynthesisAgent().process(outputs)
    crit = CritiqueAgent().process(synth)

    append_memory({
        "ticker": symbol,
        "lanes": lanes,
        "issues": crit.key_factors,
        "final_confidence": synth.confidence,
    })

    evidence = {
        "top_news": top_news.to_dict(orient="records"),
        "prices_tail": prices.tail(5).to_dict(orient="records")
    }
    return OrchestratorResult(plan, evidence, outputs, synth, crit)
