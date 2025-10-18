
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
from ..analysis.signals import fetch_indicator
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
            # Map to teammate’s keys; if you don’t have these, leave None.
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

    # # --- TECHNICALS (build the exact schema MarketSignalsAgent expects)
    # if "technical" in lanes and (has_technicals or has_prices):
    #     # basics from prices
    #     current_price = float(prices["close"].iloc[-1]) if has_prices else None
    #     volume       = int(prices["volume"].iloc[-1]) if has_prices else None
    #     avg_volume   = int(prices["volume"].tail(20).mean()) if has_prices else None

    #     # ensure you have SMA(20/50/200) available
    #     # if you already fetched them elsewhere, keep those;
    #     # otherwise compute here from prices as a fallback.
    #     sma20_df = None
    #     sma50_df = None
    #     sma200_df = None

    #     try:
    #         sma20_df = fetch_indicator(symbol, "SMA", 20)
    #     except Exception:
    #         pass
    #     try:
    #         sma50_df = fetch_indicator(symbol, "SMA", 50)
    #     except Exception:
    #         pass
    #     try:
    #         sma200_df = fetch_indicator(symbol, "SMA", 200)
    #     except Exception:
    #         pass

    #     # fallback: compute locally if fetch_indicator didn’t return
    #     if (sma20_df is None or sma20_df.empty) and has_prices:
    #         sma20_df = prices[["close"]].copy()
    #         sma20_df["SMA"] = sma20_df["close"].rolling(20, min_periods=1).mean()
    #     if (sma50_df is None or sma50_df.empty) and has_prices:
    #         sma50_df = prices[["close"]].copy()
    #         sma50_df["SMA"] = sma50_df["close"].rolling(50, min_periods=1).mean()
    #     if (sma200_df is None or sma200_df.empty) and has_prices:
    #         sma200_df = prices[["close"]].copy()
    #         sma200_df["SMA"] = sma200_df["close"].rolling(200, min_periods=1).mean()

    #     ma_20  = float(sma20_df["SMA"].iloc[-1])  if sma20_df is not None and not sma20_df.empty  else None
    #     ma_50  = float(sma50_df["SMA"].iloc[-1])  if sma50_df is not None and not sma50_df.empty  else None
    #     ma_200 = float(sma200_df["SMA"].iloc[-1]) if sma200_df is not None and not sma200_df.empty else None

    #     # RSI: you already had rsi = fetch_indicator(symbol, "RSI", 14)
    #     rsi_val = float(rsi["RSI"].iloc[-1]) if rsi is not None and not rsi.empty else None

    #     technicals = {
    #         "current_price": current_price,
    #         "rsi": rsi_val,
    #         "macd": None,        # compute later if you add it
    #         "ma_50": ma_50,      # agent expects ma_50
    #         "ma_200": ma_200,    # agent expects ma_200
    #         "volume": volume,
    #         "avg_volume": avg_volume,
    #         "support": None,     # optional
    #         "resistance": None,  # optional
    #         # optional: include ma_20 if you want, but agent doesn't need it:
    #         # "ma_20": ma_20,
    #     }

    #     outputs.append(MarketSignalsAgent().process({
    #         "ticker": symbol,
    #         "technicals": technicals
    #     }))


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

    # 6) synthesize + critique
    synth = SynthesisAgent().process(outputs)
    crit  = CritiqueAgent().process(synth)

    # 7) memory
    append_memory({
        "ticker": symbol,
        "lanes": lanes,
        "issues": crit.key_factors,
        "final_confidence": synth.confidence,
        "timestamp": datetime.utcnow().isoformat()
    })

    # 8) evidence for UI
    evidence = {
        "top_news": top_news,
        "prices_tail": prices.tail(5)
    }

    return OrchestratorResult(plan, evidence, outputs, synth, crit)
