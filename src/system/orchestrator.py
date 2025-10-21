from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from datetime import datetime, timezone
import time
import json

import pandas as pd
from pandas import DataFrame

from src.config.settings import SETTINGS
from src.data_io.prices import fetch_prices
from src.data_io.news import fetch_news
from src.data_io.indicators import fetch_indicator
from src.data_io.earnings import fetch_earnings
from src.data_io.risk import fetch_risk_metrics
from src.analysis.text import preprocess_news, add_tags_and_numbers, recent_topk
from src.system.router import choose_agents
from src.system.memory import append_memory
from src.agents import (
    NewsAnalysisAgent,
    MarketSignalsAgent,
    RiskAssessmentAgent,
    SynthesisAgent,
    CritiqueAgent,
    AgentResponse,
    EarningsAnalysisAgent,
)

# ------------- helpers -------------
def _as_text(x):
    """We coerce any object into a readable string (pretty JSON for dict/list)."""
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False, indent=2)
        except Exception:
            return str(x)
    return str(x)

def _as_list_of_text(x):
    """We normalize arbitrary input into a non-null list of strings."""
    if isinstance(x, list):
        return [_as_text(i) for i in x]
    if x is None:
        return []
    return [_as_text(x)]

def now_utc_iso() -> str:
    """We return a stable, timezone-aware timestamp for logs and memory."""
    return datetime.now(timezone.utc).isoformat()

# We use a small stagger to be kind to API rate limits.
_NET_STAGGER = float(getattr(SETTINGS, "net_stagger_secs", 0.5))

@dataclass
class OrchestratorResult:
    """We bundle everything the UI needs after one pipeline run."""
    plan: List[str]
    evidence: Dict[str, DataFrame]
    agent_outputs: List[AgentResponse]
    final: AgentResponse
    critique: AgentResponse


def run_pipeline(
    symbol: str,
    start: str | None,
    end: str | None,
    required_tags: list[str] | None = None
) -> OrchestratorResult:
    # We keep the plan visible so the UI can show progress/explain steps.
    plan = [
        "fetch_prices", "fetch_news", "fetch_earnings", "fetch_risk",
        "preprocess", "classify_extract", "retrieve_topk",
        "route", "run_agents", "synthesize", "critique", "save_memory"
    ]

    # ---------------- 1) FETCH (staggered) ----------------
    prices = fetch_prices(symbol, start, end);                      time.sleep(_NET_STAGGER)
    news   = fetch_news(symbol);                                     time.sleep(_NET_STAGGER)
    earn_df = fetch_earnings(symbol);                                time.sleep(_NET_STAGGER)
    risk_ingested = fetch_risk_metrics(symbol, start, end);          time.sleep(_NET_STAGGER)

    # ---------------- 2) PREPROCESS NEWS ----------------
    # We clean the articles and add lightweight tags/numbers for filtering.
    news_pp = add_tags_and_numbers(preprocess_news(news))

    # ---------------- 3) RETRIEVAL ----------------
    # We keep a small, recent slice for agents to read.
    top_news = recent_topk(
        news_pp,
        topk=SETTINGS.topk_news,
        days=SETTINGS.news_window_days,
        required_tags=required_tags,
    )

    # ---------------- 4) ROUTE PRIMERS ----------------
    has_news     = not top_news.empty
    has_prices   = not prices.empty
    has_earnings = (earn_df is not None) and (not earn_df.empty)

    # We try indicators if we have prices or a provider key.
    attempt_technicals = has_prices or bool(SETTINGS.alpha_api_key)

    # ---------------- 5) INDICATORS (conditional) ----------------
    rsi = sma20 = sma50 = sma200 = pd.DataFrame()
    if attempt_technicals:
        rsi    = fetch_indicator(symbol, "RSI", 14); time.sleep(_NET_STAGGER)
        sma20  = fetch_indicator(symbol, "SMA", 20); time.sleep(_NET_STAGGER)
        sma50  = fetch_indicator(symbol, "SMA", 50); time.sleep(_NET_STAGGER)
        sma200 = fetch_indicator(symbol, "SMA", 200); time.sleep(_NET_STAGGER)

    has_technicals = (not rsi.empty) or (not sma20.empty) or (not sma50.empty) or (not sma200.empty)

    # We let the router decide which lanes to run (news/technical/earnings/risk).
    lanes = choose_agents(has_news, has_prices, has_technicals, has_earnings)

    # ---------------- 6) RUN AGENTS ----------------
    outputs: List[AgentResponse] = []

    # NEWS
    if "news" in lanes and has_news:
        # We map to the keys the NewsAnalysisAgent expects.
        news_payload_records = (
            top_news
            .rename(columns={"summary": "description"})
            .loc[:, ["title", "description", "source", "url", "published_at"]]
            .to_dict(orient="records")
        )
        outputs.append(NewsAnalysisAgent().process({"ticker": symbol, "news": news_payload_records}))

    # TECHNICALS
    if "technical" in lanes and (has_technicals or has_prices):
        # We compute a tiny snapshot of technical state.
        current_price = float(prices["close"].iloc[-1]) if has_prices else None
        volume = int(prices["volume"].iloc[-1]) if has_prices else None
        avg_volume = int(prices["volume"].tail(20).mean()) if has_prices else None

        technicals = {
            "current_price": current_price,
            "rsi": (float(rsi["RSI"].iloc[-1]) if not rsi.empty else None),
            "ma_50": (float(sma50["SMA"].iloc[-1]) if not sma50.empty else
                      (float(sma20["SMA"].iloc[-1]) if not sma20.empty else None)),
            "ma_200": (float(sma200["SMA"].iloc[-1]) if not sma200.empty else None),
            "macd": None,      # reserved for future
            "volume": volume,
            "avg_volume": avg_volume,
            "support": None,   # reserved for future
            "resistance": None # reserved for future
        }
        outputs.append(MarketSignalsAgent().process({"ticker": symbol, "technicals": technicals}))

    # EARNINGS
    if "earnings" in lanes and has_earnings:
        earn_payload = {
            "ticker": symbol,
            "earnings": (
                earn_df.sort_values("date", ascending=False)
                       .head(8)
                       .to_dict(orient="records")
            )
        }
        outputs.append(EarningsAnalysisAgent().process(earn_payload))

    # RISK (merge ingestion + a quick realized 30d vol for the UI)
    vol_30d = float(prices["close"].pct_change().tail(30).std() * 100) if has_prices else None
    risk_payload = {
        "ticker": symbol,
        "risk_metrics": {
            "beta":              risk_ingested.get("beta"),
            "volatility":        vol_30d,                        # short-term display (%)
            "var_5":             risk_ingested.get("var_5"),
            "sharpe_ratio":      risk_ingested.get("sharpe_ratio"),
            "max_drawdown":      risk_ingested.get("max_drawdown"),
            "sector_correlation": None,
            "pe_ratio":          None,
            "avg_daily_return":  risk_ingested.get("avg_daily_return"),
            "volatility_full":   risk_ingested.get("volatility"),
        }
    }
    outputs.append(RiskAssessmentAgent().process(risk_payload))

    # ---------------- 7) SYNTHESIZE + CRITIQUE ----------------
    synth_v1 = SynthesisAgent().process(outputs)     # first pass
    crit     = CritiqueAgent().process(synth_v1)     # critique of first pass

    # We gate synth_v2 behind a simple rule to avoid unnecessary extra calls.
    needs_rerun = (crit.score < 0.90) or (
        "data quality" in " ".join(_as_list_of_text(crit.key_factors)).lower()
    )

    synth_final = synth_v1  # default to v1 unless we improve it
    synth_v2 = None         # we keep a handle for telemetry/UI if needed

    if needs_rerun:
        # We turn the critique into an explicit feedback message the synthesizer can read.
        critique_feedback = AgentResponse(
            agent_name="Critique Feedback",
            analysis=_as_text(synth_v1.analysis) + "\n\n[CRITIQUE]\n" + _as_text(crit.analysis),
            score=crit.score,
            confidence=crit.confidence,
            key_factors=_as_list_of_text(crit.key_factors),
            timestamp=now_utc_iso()
        )
        # We re-run synthesis with the feedback appended to the agent outputs.
        synth_v2_inputs = outputs + [critique_feedback]
        synth_v2 = SynthesisAgent().process(synth_v2_inputs)
        synth_final = synth_v2

    # ---------------- 8) MEMORY ----------------
    # We store whether the optimizer path ran and both confidences for later review.
    append_memory({
        "ticker": symbol,
        "lanes": lanes,
        "issues": crit.key_factors,
        "final_confidence_v1": synth_v1.confidence,
        "final_confidence_v2": (synth_v2.confidence if synth_v2 else None),
        "optimizer_triggered": bool(needs_rerun),
        "timestamp": now_utc_iso()
    })

    # ---------------- 9) EVIDENCE FOR UI ----------------
    earn_evidence = (
        earn_df.sort_values("date", ascending=False).head(8)
        if has_earnings else pd.DataFrame()
    )
    risk_evidence = pd.DataFrame([risk_payload["risk_metrics"]])

    evidence = {
        "top_news": top_news,
        "prices_tail": prices.tail(5),
        "earnings_head": earn_evidence,
        "risk_metrics": risk_evidence,
    }

    # We add the initial synthesis as a separate output so the UI can compare v1 vs final.
    outputs.append(AgentResponse(
        agent_name="Initial Synthesis",
        analysis=_as_text(synth_v1.analysis),
        score=float(synth_v1.score),
        confidence=float(synth_v1.confidence),
        key_factors=_as_list_of_text(synth_v1.key_factors),
        timestamp=synth_v1.timestamp
    ))

    return OrchestratorResult(plan, evidence, outputs, synth_final, crit)
