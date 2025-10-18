
# ======= 3Rd version of the orchestrator ==========

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
from datetime import datetime
import pandas as pd
from pandas import DataFrame

from ..config.settings import SETTINGS
from ..data_io.prices import fetch_prices
from ..data_io.news import fetch_news
from ..data_io.indicators import fetch_indicator
from ..analysis.text import preprocess_news, add_tags_and_numbers, recent_topk
from ..system.router import choose_agents
from ..system.memory import append_memory
from ..agents import (
    NewsAnalysisAgent,
    MarketSignalsAgent,
    RiskAssessmentAgent,
    SynthesisAgent,
    CritiqueAgent,
    AgentResponse,
)

# from ..analysis.signals import normalize_technicals  # add this import

import json

def _as_text(x):
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False, indent=2)
        except Exception:
            return str(x)
    return str(x)

def _as_list_of_text(x):
    if isinstance(x, list):
        return [_as_text(i) for i in x]
    if x is None:
        return []
    # if a single str/dict was returned, wrap it
    return [_as_text(x)]


@dataclass
class OrchestratorResult:
    plan: List[str]
    evidence: Dict[str, DataFrame]
    agent_outputs: List[AgentResponse]
    final: AgentResponse
    critique: AgentResponse


def run_pipeline(symbol: str, start: str | None, end: str | None,
                 required_tags: list[str] | None = None) -> OrchestratorResult:
    plan = [
        "fetch_prices", "fetch_news", "preprocess", "classify_extract",
        "retrieve_topk", "route", "run_agents", "synthesize", "critique", "save_memory"
    ]

    # 1) fetch
    prices = fetch_prices(symbol, start, end)
    news   = fetch_news(symbol)

    # 2) preprocess (clean + tags + lightweight numeric extraction)
    news_pp = add_tags_and_numbers(preprocess_news(news))

    # 3) retrieval (deterministic)
    top_news = recent_topk(
        news_pp,
        topk=SETTINGS.topk_news,
        days=SETTINGS.news_window_days,
        required_tags=required_tags
    )

    # 4) route
    has_news   = not top_news.empty
    has_prices = not prices.empty

    # Technical indicators (fetch what we can; leave the rest as None)
    rsi = fetch_indicator(symbol, "RSI", 14)         # may be empty
    sma20 = fetch_indicator(symbol, "SMA", 20)       # may be empty
    # Optional (if your fetch_indicator supports these; otherwise they'll be empty DataFrames)
    sma50 = fetch_indicator(symbol, "SMA", 50)
    sma200 = fetch_indicator(symbol, "SMA", 200)

    has_technicals = (not rsi.empty) or (not sma20.empty) or (not sma50.empty) or (not sma200.empty)
    lanes = choose_agents(has_news, has_prices, has_technicals)

    # 5) run agents
    outputs: List[AgentResponse] = []

    # NEWS (delegate summarization to the agent)
    if "news" in lanes and has_news:
        # Teammate agent expects `description`; remap our `summary` to `description`
        news_payload_records = (
            top_news
            .rename(columns={"summary": "description"})
            .loc[:, ["title", "description", "source", "url", "published_at"]]
            .to_dict(orient="records")
        )
        news_payload = {
            "ticker": symbol,
            "news": news_payload_records
        }
        outputs.append(NewsAnalysisAgent().process(news_payload))

    # TECHNICALS (build dict the agent expects; missing fields are fine)
    if "technical" in lanes and (has_technicals or has_prices):
        current_price = float(prices["close"].iloc[-1]) if has_prices else None
        volume = int(prices["volume"].iloc[-1]) if has_prices else None
        avg_volume = int(prices["volume"].tail(20).mean()) if has_prices else None

        technicals = {
            "current_price": current_price,
            "rsi": (float(rsi["RSI"].iloc[-1]) if not rsi.empty else None),
            # Map to teammate’s keys; if we don’t have these, leave None.
            "ma_50": (float(sma50["SMA"].iloc[-1]) if not sma50.empty else (float(sma20["SMA"].iloc[-1]) if not sma20.empty else None)),
            "ma_200": (float(sma200["SMA"].iloc[-1]) if not sma200.empty else None),
            "macd": None,                # not computed here
            "volume": volume,
            "avg_volume": avg_volume,
            "support": None,
            "resistance": None,
        }
        tech_payload = {
            "ticker": symbol,
            "technicals": technicals
        }
        outputs.append(MarketSignalsAgent().process(tech_payload))


    # RISK (simple proxy; fill what you have)
    vol_30d = float(prices["close"].pct_change().tail(30).std() * 100) if has_prices else None
    risk_payload = {
        "ticker": symbol,
        "risk_metrics": {
            "beta": None,
            "volatility": vol_30d,      # teammate agent expects 'volatility' (%)
            "var_5": None,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "sector_correlation": None,
            "pe_ratio": None
        }
    }
    outputs.append(RiskAssessmentAgent().process(risk_payload))

    # 6) synthesize + critique (gated second pass)
    synth_v1 = SynthesisAgent().process(outputs)
    crit     = CritiqueAgent().process(synth_v1)

    # Decide if we need a second pass.
    # Rule: re-run if critique quality score < 0.90 OR if it mentions "data quality".
    needs_rerun = (crit.score < 0.90) or ("data quality" in " ".join(crit.key_factors).lower())

    synth_final = synth_v1
    if needs_rerun:
        # Turn critique into feedback for the optimizer pass
        critique_feedback = AgentResponse(
            agent_name="Critique Feedback",
            analysis=_as_text(synth_v1.analysis) + "\n\n[CRITIQUE]\n" + _as_text(crit.analysis),
            score=crit.score,
            confidence=crit.confidence,
            key_factors=_as_list_of_text(crit.key_factors),
            timestamp=datetime.utcnow().isoformat()
            
        )

        synth_v2_inputs = outputs + [critique_feedback]
        synth_v2 = SynthesisAgent().process(synth_v2_inputs)
        synth_final = synth_v2

    # 7) memory (store both passes where applicable)
    append_memory({
        "ticker": symbol,
        "lanes": lanes,
        "issues": crit.key_factors,
        "final_confidence_v1": synth_v1.confidence,
        "final_confidence_v2": synth_final.confidence if needs_rerun else None,
        "optimizer_triggered": bool(needs_rerun),
        "timestamp": datetime.utcnow().isoformat()
    })

    # 8) evidence for UI
    evidence = {
        "top_news": top_news,
        "prices_tail": prices.tail(5)
    }

    # Show the first synthesis alongside other agents for transparency

    outputs.append(AgentResponse(
        agent_name="Initial Synthesis",
        analysis=_as_text(synth_v1.analysis),
        score=float(synth_v1.score),
        confidence=float(synth_v1.confidence),
        key_factors=_as_list_of_text(synth_v1.key_factors),
        timestamp=synth_v1.timestamp
))

    return OrchestratorResult(plan, evidence, outputs, synth_final, crit)

