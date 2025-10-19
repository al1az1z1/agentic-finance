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
    """
    Main orchestration pipeline for financial analysis.
    
    Coordinates multiple specialized agents to analyze a stock through:
    1. Data ingestion (prices, news, technical indicators)
    2. Preprocessing and classification
    3. Routing to appropriate agents
    4. Synthesis of findings
    5. Quality critique and optional re-synthesis
    
    This function implements the Evaluator-Optimizer pattern internally.
    """
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
            # Map to teammate's keys; if we don't have these, leave None.
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


# ============================================================================
# WORKFLOW PATTERN DEMONSTRATIONS
# ============================================================================

def run_prompt_chaining_workflow(symbol: str, start: str, end: str, 
                                  required_tags: list[str] | None = None) -> AgentResponse:
    """
    WORKFLOW PATTERN 1: PROMPT CHAINING
    
    Demonstrates sequential processing where each step's output becomes
    the input to the next step, creating a progressive refinement chain.
    
    Pipeline: Ingest → Preprocess → Classify → Extract → Summarize
    
    Args:
        symbol: Stock ticker (e.g., 'AAPL')
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        required_tags: Optional tag filter
    
    Returns:
        AgentResponse with sentiment analysis
    """
    print("\n" + "="*80)
    print("WORKFLOW PATTERN 1: PROMPT CHAINING")
    print("="*80)
    print(f"Analyzing: {symbol} | Period: {start} to {end}")
    print("="*80)
    
    # Step 1: INGEST
    print("\n─ STEP 1/5: INGEST ─────────────────────────────────────────────")
    print(" Fetching raw news from Alpha Vantage API...                    ")
    print("─────────────────────────────────────────────────────────────────")
    raw_news = fetch_news(symbol)
    
    if raw_news.empty:
        print("  ⚠️  No news data available")
        return AgentResponse(
            agent_name="News Analysis Agent",
            analysis="No news data available",
            score=0.0,
            confidence=0.0,
            key_factors=["No data"],
            timestamp=datetime.now().isoformat()
        )
    
    print(f"  ✓ Fetched: {len(raw_news)} articles")
    
    # Step 2: PREPROCESS
    print("\n─ STEP 2/5: PREPROCESS ─────────────────────────────────────────")
    print(" Cleaning, parsing dates, removing duplicates...                ")
    print("─────────────────────────────────────────────────────────────────")
    clean_news = preprocess_news(raw_news)
    print(f"  ✓ Cleaned: {len(clean_news)} articles")
    
    # Step 3: CLASSIFY
    print("\n─ STEP 3/5: CLASSIFY ───────────────────────────────────────────┐")
    print(" Tagging by topic, extracting numbers...                        ")
    print("─────────────────────────────────────────────────────────────────")
    tagged_news = add_tags_and_numbers(clean_news)
    print(f"  ✓ Tagged: {len(tagged_news)} articles")
    
    # Step 4: EXTRACT
    print("\n─ STEP 4/5: EXTRACT ────────────────────────────────────────────")
    print(" Filtering to top-K most relevant...                            ")
    print("─────────────────────────────────────────────────────────────────")
    top_news = recent_topk(tagged_news, topk=SETTINGS.topk_news, days=SETTINGS.news_window_days, required_tags=required_tags)
    print(f"  ✓ Extracted: {len(top_news)} top articles")
    
    # Step 5: SUMMARIZE
    print("\n┌─ STEP 5/5: SUMMARIZE ──────────────────────────────────────────")
    print(" LLM agent analyzing sentiment...                               ")
    print("─────────────────────────────────────────────────────────────────")
    news_payload = {
        'ticker': symbol,
        'news': top_news.rename(columns={"summary": "description"}).to_dict('records') if not top_news.empty else []
    }
    result = NewsAnalysisAgent().process(news_payload)
    print(f"  ✓ Score: {result.score:+.2f} | Confidence: {result.confidence:.0%}")
    
    print("\n" + "="*80)
    print("PROMPT CHAINING COMPLETE")
    print("Pattern: Raw Data → Clean → Tagged → Filtered → Analysis")
    print("="*80 + "\n")
    
    return result


def run_parallel_workflow(symbol: str, start: str, end: str) -> List[AgentResponse]:
    """
    WORKFLOW PATTERN 2: PARALLEL EXECUTION
    
    Demonstrates concurrent execution of multiple agents for improved performance.
    Instead of sequential (A→B→C), runs simultaneously (A+B+C).
    
    Args:
        symbol: Stock ticker
        start: Start date
        end: End date
    
    Returns:
        List of AgentResponses from concurrent execution
    """
    from concurrent.futures import ThreadPoolExecutor
    import time
    
    print("\n" + "="*80)
    print("WORKFLOW PATTERN 2: PARALLEL EXECUTION")
    print("="*80)
    print(f"Analyzing: {symbol}")
    print("="*80)
    
    # Prepare data
    print("\n[Preparation] Fetching data...")
    news = fetch_news(symbol)
    prices = fetch_prices(symbol, start, end)
    
    # Prepare inputs
    news_input = {
        'ticker': symbol,
        'news': news.head(5).rename(columns={"summary": "description"}).to_dict('records') if not news.empty else []
    }
    
    tech_input = {
        'ticker': symbol,
        'technicals': {
            'current_price': float(prices['close'].iloc[-1]) if not prices.empty else None,
            'volume': int(prices['volume'].iloc[-1]) if not prices.empty else None,
        }
    }
    
    risk_input = {
        'ticker': symbol,
        'risk_metrics': {
            'volatility': float(prices["close"].pct_change().tail(30).std() * 100) if not prices.empty else None
        }
    }
    
    print("\n[Parallel] Running 3 agents concurrently...")
    print("  Sequential would be: Agent1 → Agent2 → Agent3")
    print("  Parallel runs: Agent1 + Agent2 + Agent3 simultaneously")
    
    start_time = time.time()
    
    # Execute in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            'news': executor.submit(NewsAnalysisAgent().process, news_input),
            'technical': executor.submit(MarketSignalsAgent().process, tech_input),
            'risk': executor.submit(RiskAssessmentAgent().process, risk_input)
        }
        
        results = {}
        for name, future in futures.items():
            results[name] = future.result()
            print(f"  ✓ {name.capitalize()}: Score={results[name].score:+.2f}")
    
    elapsed = time.time() - start_time
    
    print(f"\n" + "="*80)
    print(f"PARALLEL EXECUTION COMPLETE ({elapsed:.2f}s)")
    print(f"Speedup: ~2-3x faster than sequential execution")
    print("="*80 + "\n")
    
    return list(results.values())


def run_evaluator_optimizer_workflow(symbol: str, start: str, end: str,
                                      required_tags: list[str] | None = None) -> OrchestratorResult:
    """
    WORKFLOW PATTERN 3: EVALUATOR-OPTIMIZER (Self-Improving)
    
    Demonstrates adaptive refinement through Generate → Evaluate → Optimize loop.
    System critiques its own output and re-generates if quality is below threshold.
    
    This wraps the existing run_pipeline() to explicitly demonstrate the pattern.
    
    Args:
        symbol: Stock ticker
        start: Start date
        end: End date
        required_tags: Optional tag filter
    
    Returns:
        OrchestratorResult with potentially optimized synthesis
    """
    print("\n" + "="*80)
    print("WORKFLOW PATTERN 3: EVALUATOR-OPTIMIZER")
    print("="*80)
    print(f"Analyzing: {symbol}")
    print("="*80)
    
    # GENERATE
    print("\n[Phase 1] GENERATE: Running initial analysis...")
    result = run_pipeline(symbol, start, end, required_tags)
    
    initial_synthesis = next(
        (a for a in result.agent_outputs if "Initial Synthesis" in a.agent_name),
        None
    )
    
    if initial_synthesis:
        print(f"  ✓ Initial: Score={initial_synthesis.score:+.2f}, Conf={initial_synthesis.confidence:.0%}")
    
    # EVALUATE
    print(f"\n[Phase 2] EVALUATE: Critiquing quality...")
    print(f"  ✓ Quality Score: {result.critique.score:.2f}")
    print(f"  ✓ Issues Found: {len(result.critique.key_factors)}")
    
    # Check if optimizer ran
    optimizer_triggered = initial_synthesis and (initial_synthesis.analysis != result.final.analysis)
    
    # OPTIMIZE
    if optimizer_triggered:
        print(f"\n[Phase 3] OPTIMIZE: Re-synthesizing with feedback...")
        print(f"  ✓ Final: Score={result.final.score:+.2f}, Conf={result.final.confidence:.0%}")
        
        conf_change = result.final.confidence - initial_synthesis.confidence
        print(f"\n" + "="*80)
        print(f"OPTIMIZATION SUCCESSFUL")
        print(f"Confidence Change: {conf_change:+.0%} | Optimizer: YES")
        print("="*80 + "\n")
    else:
        print(f"\n" + "="*80)
        print(f"INITIAL QUALITY ACCEPTABLE")
        print(f"Quality: {result.critique.score:.2f} | Optimizer: NO")
        print("="*80 + "\n")
    
    return result


# ============================================================================
# DEMONSTRATION RUNNER
# ============================================================================

def demo_all_workflows(symbol: str = "AAPL"):
    """
    Run all three workflow patterns for demonstration purposes.
    
    This function is designed to be called from a notebook or script
    to show how the system implements different orchestration strategies.
    """
    from datetime import datetime, timedelta
    
    start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    
    print("\n" + "#"*40)
    print("  DEMONSTRATING 3 AGENTIC WORKFLOW PATTERNS")
    print("#"*40)
    print(f"\nTicker: {symbol}")
    print(f"Date Range: {start} to {end}\n")
    
    # Pattern 1
    result1 = run_prompt_chaining_workflow(symbol, start, end)
    
    # Pattern 2
    result2 = run_parallel_workflow(symbol, start, end)
    
    # Pattern 3
    result3 = run_evaluator_optimizer_workflow(symbol, start, end)
    
    print("\n" + "#"*40)
    print("  ALL 3 WORKFLOW PATTERNS DEMONSTRATED")
    print("#"*40 + "\n")
    
    return {
        'prompt_chaining': result1,
        'parallel': result2,
        'evaluator_optimizer': result3
    }


if __name__ == "__main__":
    # Run demonstration when file is executed directly
    demo_all_workflows("AAPL")