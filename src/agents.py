from __future__ import annotations
import os, json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

#region added for Weird ScoreScakes 
import re

def strip_code_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    # remove leading/trailing ``` blocks
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)

def to_float(x, default=0.0):
    try:
        if isinstance(x, str):
            x = x.strip()
            # common words to numeric
            if x.lower() in ("high", "strong", "bullish", "overbought"): 
                return 0.8
            if x.lower() in ("medium", "moderate", "neutral"):
                return 0.5
            if x.lower() in ("low", "weak", "bearish", "oversold"):
                return 0.2
        v = float(x)
        return v
    except Exception:
        return default

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def normalize_score(v: float) -> float:
    """
    Map various score ranges to [-1, 1] for agent 'score' fields.
    Heuristics:
      - If already in [-1,1], keep.
      - If in [0,1], map to [-1,1] via (v-0.5)*2.
      - If in [0,100], divide by 100 then map.
      - Else, clamp.
    """
    if -1.0 <= v <= 1.0:
        return v
    if 0.0 <= v <= 1.0:
        return (v - 0.5) * 2.0
    if 1.0 < v <= 100.0:
        v01 = v / 100.0
        return (v01 - 0.5) * 2.0
    # weird values (e.g., 8.5), assume 0-10
    if 1.0 < v <= 10.0:
        v01 = v / 10.0
        return (v01 - 0.5) * 2.0
    return clamp(v, -1.0, 1.0)

def normalize_conf(v) -> float:
    f = to_float(v, 0.7)
    # map 0-100 to 0-1 if needed
    if f > 1.0 and f <= 100.0:
        f = f / 100.0
    return clamp(f, 0.0, 1.0)
#endregion added for Weird ScoreScakes 


OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    from openai import OpenAI
    _client = OpenAI(api_key=OPENAI_KEY)
else:
    _client = None  # mock mode

@dataclass
class AgentResponse:
    agent_name: str
    analysis: str
    score: float
    confidence: float
    key_factors: List[str]
    timestamp: str

class BaseAgent:
    def __init__(self, agent_name: str, model: str = "gpt-4o-mini"):
        self.agent_name = agent_name
        self.model = model

    def call_llm(self, system_prompt: str, user_message: str) -> str:
        if _client is None:  # mock
            return json.dumps({"analysis": f"MOCK: {self.agent_name} processed.", "score": 0.0,
                               "key_factors": ["mock"], "confidence": 0.7})
        try:
            resp = _client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":system_prompt},
                          {"role":"user","content":user_message}],
                temperature=0.3, max_tokens=800
            )
            return resp.choices[0].message.content
        except Exception as e:
            return json.dumps({"analysis": f"Error: {e}", "score": 0.0, "key_factors": ["error"], "confidence": 0.3})

# class NewsAnalysisAgent(BaseAgent):
#     def __init__(self, model: str = "gpt-4o-mini"):
#         super().__init__("News Analysis Agent", model)
#     sys = ("You are a financial news analyst. Return JSON with keys: "
#            "sentiment_score, analysis, key_factors, confidence.")
#     def process(self, ticker: str, bullets: List[str]) -> AgentResponse:
#         msg = f"Ticker {ticker}. News bullets:\n" + "\n".join(bullets[:5])
       
#         out = self.call_llm(self.sys, msg)
#         js = strip_code_fences(out)
#         try:
#             j = json.loads(js)
#         except json.JSONDecodeError:
#             j = {"sentiment_score": 0, "analysis": out, "key_factors": ["parse_fail"], "confidence": 0.5}

#         sent = to_float(j.get("sentiment_score", 0.0), 0.0)
#         score = normalize_score(sent)  # to [-1, 1]
#         conf  = normalize_conf(j.get("confidence", 0.7))
#         kf    = j.get("key_factors", [])
#         return AgentResponse("News Analysis Agent", j.get("analysis",""), score, conf, kf, datetime.now().isoformat())




# class MarketSignalsAgent(BaseAgent):
#     def __init__(self, model: str = "gpt-4o-mini"):
#         super().__init__("Market Signals Agent", model)
#     sys = ("You are a technical analyst. Return JSON with keys: "
#            "technical_score, analysis, key_factors, confidence.")
#     def process(self, ticker: str, technicals: Dict[str, Any]) -> AgentResponse:
#         msg = f"Ticker {ticker}. Technicals: {technicals}"

#         out = self.call_llm(self.sys, msg)
#         js = strip_code_fences(out)
#         try:
#             j = json.loads(js)
#         except json.JSONDecodeError:
#             j = {"technical_score": 0, "analysis": out, "key_factors": ["parse_fail"], "confidence": 0.5}

#         tech = to_float(j.get("technical_score", 0.0), 0.0)
#         score = normalize_score(tech)  # to [-1, 1]
#         conf  = normalize_conf(j.get("confidence", 0.7))
#         kf    = j.get("key_factors", [])
#         return AgentResponse("Market Signals Agent", j.get("analysis",""), score, conf, kf, datetime.now().isoformat())


# class RiskAssessmentAgent(BaseAgent):
#     def __init__(self, model: str = "gpt-4o-mini"):
#         super().__init__("Risk Assessment Agent", model)
#     sys = ("You are a risk analyst. Return JSON with keys: "
#            "risk_score, analysis, key_factors, confidence.")
#     def process(self, ticker: str, risk: Dict[str, Any]) -> AgentResponse:
#         msg = f"Ticker {ticker}. Risk metrics: {risk}"

#         out = self.call_llm(self.sys, msg)
#         js = strip_code_fences(out)
#         try:
#             j = json.loads(js)
#         except json.JSONDecodeError:
#             j = {"risk_score": 0.5, "analysis": out, "key_factors": ["parse_fail"], "confidence": 0.6}

#         risk01 = to_float(j.get("risk_score", 0.5), 0.5)
#         # normalize risk to 0..1 if given as 0..100 or 0..10
#         if risk01 > 1.0 and risk01 <= 100.0:
#             risk01 = risk01 / 100.0
#         elif risk01 > 1.0 and risk01 <= 10.0:
#             risk01 = risk01 / 10.0
#         risk01 = clamp(risk01, 0.0, 1.0)

#         # convert to signed score where higher risk => more negative
#         score = -(risk01 - 0.5) * 2.0
#         score = clamp(score, -1.0, 1.0)

#         conf  = normalize_conf(j.get("confidence", 0.7))
#         kf    = j.get("key_factors", [])
#         return AgentResponse("Risk Assessment Agent", j.get("analysis",""), score, conf, kf, datetime.now().isoformat())


# class SynthesisAgent(BaseAgent):
#     def __init__(self, model: str = "gpt-4o-mini"):
#         super().__init__("Research Synthesis Agent", model)
#     sys = ("You are a senior investment analyst. Return JSON with keys: "
#            "recommendation, confidence, analysis, key_points, risks.")
#     def process(self, parts: List[AgentResponse]) -> AgentResponse:
#         blob = "\n\n".join([f"{p.agent_name} | score={p.score} | {p.analysis}" for p in parts])

#         out = self.call_llm(self.sys, f"Synthesize:\n{blob}")
#         js = strip_code_fences(out)
#         try:
#             j = json.loads(js)
#             rec2score = {"STRONG BUY": 1.0, "BUY": 0.6, "HOLD": 0.0, "SELL": -0.6, "STRONG SELL": -1.0}
#             score = rec2score.get(str(j.get("recommendation","HOLD")).upper(), 0.0)
#             conf  = normalize_conf(j.get("confidence", 0.7))
#             kf = j.get("key_points", []) + j.get("risks", [])
#             return AgentResponse("Research Synthesis Agent", j.get("analysis",""), float(score), conf, kf, datetime.now().isoformat())
#         except json.JSONDecodeError:
#             return AgentResponse("Research Synthesis Agent", out, 0.0, 0.6, ["parse_fail"], datetime.now().isoformat())


# class CritiqueAgent(BaseAgent):
#     def __init__(self, model: str = "gpt-4o-mini"):
#         super().__init__("Critique & Validation Agent", model)
#     sys = ("You are a critique analyst. Return JSON with keys: "
#            "quality_score, issues_found, suggestions, adjusted_confidence.")
#     def process(self, synthesis: AgentResponse) -> AgentResponse:

#         out = self.call_llm(self.sys, f"Review: {synthesis.analysis}")
#         js = strip_code_fences(out)
#         try:
#             j = json.loads(js)
#             q  = to_float(j.get("quality_score", 0.7), 0.7)
#             # quality is display-only; map 0..10 to 0..1 for score if >1
#             q_score = q/10.0 if q > 1.0 and q <= 10.0 else (q/100.0 if q > 10.0 else q)
#             q_score = clamp(q_score, 0.0, 1.0)
#             adj = normalize_conf(j.get("adjusted_confidence", synthesis.confidence))
#             text = f"Quality={j.get('quality_score',0.7)} | Issues={j.get('issues_found',[])} | Suggs={j.get('suggestions',[])}"
#             return AgentResponse("Critique & Validation Agent", text, q_score, adj, j.get("issues_found",[]), datetime.now().isoformat())
#         except json.JSONDecodeError:
#             return AgentResponse("Critique & Validation Agent", out, 0.7, synthesis.confidence, ["parse_fail"], datetime.now().isoformat())




# ====== Second version without comments ======


class NewsAnalysisAgent(BaseAgent):
    """Analyzes financial news sentiment and impact"""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("News Analysis Agent", model)
        self.system_prompt = """You are a financial news analyst specializing in sentiment analysis.
Analyze news articles about companies and provide:
1. Overall sentiment score (-1 to +1, where -1 is very negative, 0 is neutral, +1 is very positive)
2. Key factors driving the sentiment
3. Potential impact on stock price

Be objective and consider both positive and negative aspects.
Return response in JSON format with keys: sentiment_score, analysis, key_factors, confidence"""

    def process(self, data: Dict[str, Any]) -> AgentResponse:
        ticker = data.get('ticker', 'AAPL')
        news_articles = data.get('news', [])

        news_summary = "\n".join([
            f"- {a.get('title','')}: {a.get('description') or a.get('summary','')}"
            for a in news_articles[:5]
        ])

        user_message = f"""Analyze the following recent news about {ticker}:

{news_summary}

Provide sentiment analysis and impact assessment."""

        raw = self.call_llm(self.system_prompt, user_message)
        js  = strip_code_fences(raw)

        try:
            result = json.loads(js)
            score = normalize_score(to_float(result.get('sentiment_score', 0), 0.0))
            analysis = result.get('analysis', raw)
            key_factors = result.get('key_factors', [])
            confidence = normalize_conf(result.get('confidence', 0.7))
        except json.JSONDecodeError:
            score = 0.0
            analysis = raw
            key_factors = ["Unable to parse structured response"]
            confidence = 0.6

        return AgentResponse(
            agent_name=self.agent_name,
            analysis=analysis,
            score=float(score),
            confidence=float(confidence),
            key_factors=key_factors,
            timestamp=datetime.now().isoformat()
        )

# #====== Forth approach ======
# class MarketSignalsAgent(BaseAgent):
#     """Performs technical analysis on market data"""

#     def __init__(self, model: str = "gpt-4o-mini"):
#         super().__init__("Market Signals Agent", model)
#         self.system_prompt = """You are a technical analyst specializing in market signals and price patterns.
# Analyze technical indicators and provide:
# 1. Technical strength score (-1 to +1, where -1 is very bearish, +1 is very bullish)
# 2. Key technical indicators assessment
# 3. Support and resistance levels
# 4. Trend analysis

# Return response in JSON format with keys: technical_score, analysis, key_factors, confidence"""

#     def process(self, data: Dict[str, Any]) -> AgentResponse:
#         ticker = data.get('ticker', 'UNKNOWN')
#         technicals = data.get('technicals', {})

#         technical_summary = f"""
# Ticker: {ticker}
# Current Price: ${technicals.get('current_price', 'N/A')}
# 50-day MA: ${technicals.get('ma_50', 'N/A')}
# 200-day MA: ${technicals.get('ma_200', 'N/A')}
# RSI: {technicals.get('rsi', 'N/A')}
# MACD: {technicals.get('macd', 'N/A')}
# Volume: {technicals.get('volume', 'N/A')} (Avg: {technicals.get('avg_volume', 'N/A')})
# Support: ${technicals.get('support', 'N/A')}
# Resistance: ${technicals.get('resistance', 'N/A')}
# """

#         user_message = f"""Analyze the following technical data for {ticker}:

# {technical_summary}

# Assess technical strength and price momentum."""

#         raw = self.call_llm(self.system_prompt, user_message)
#         js  = strip_code_fences(raw)

#         try:
#             result = json.loads(js)
#             score = normalize_score(to_float(result.get('technical_score', 0), 0.0))
#             analysis = result.get('analysis', raw)
#             key_factors = result.get('key_factors', [])
#             confidence = normalize_conf(result.get('confidence', 0.7))
#         except json.JSONDecodeError:
#             score = 0.0
#             analysis = raw
#             key_factors = ["Unable to parse structured response"]
#             confidence = 0.6

#         return AgentResponse(
#             agent_name=self.agent_name,
#             analysis=analysis,
#             score=float(score),
#             confidence=float(confidence),
#             key_factors=key_factors,
#             timestamp=datetime.now().isoformat()
#         )

class MarketSignalsAgent(BaseAgent):
    """Performs technical analysis on market data"""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("Market Signals Agent", model)
        # 🔧 Revised prompt: stricter schema + no echoing raw dict
#         self.system_prompt = """You are a technical analyst specializing in market signals and price patterns.
# Analyze technical indicators and provide:
# 1) Technical strength score (-1 to +1, where -1 is very bearish and +1 is very bullish)
# 2) Key technical indicators assessment
# 3) Support and resistance levels
# 4) Trend analysis

# Return your answer strictly as JSON with these keys:
# - technical_score: float in [-1, 1]
# - analysis: 4–6 sentences of prose. Do NOT echo or reprint the input dictionary; write a human-readable explanation instead.
# - key_factors: list of 3–6 short bullet phrases (strings) capturing the drivers of the score
# - confidence: float in [0, 1]

# Do not include any other keys. Do not wrap the JSON in code fences."""
        self.system_prompt = """You are a technical analyst specializing in market signals and price patterns.
Analyze technical indicators and provide:
1) Technical strength score (-1 to +1, where -1 is very bearish and +1 is very bullish)
2) Key technical indicators assessment
3) Support and resistance levels
4) Trend analysis

Return your answer strictly as JSON with these keys:
- technical_score: float in [-1, 1]
- analysis: 4–6 sentences of prose that INCLUDE the exact input numbers you used (price, MAs, RSI, volume & avg). Do NOT echo the raw dict; write a human-readable explanation with those figures.
- key_factors: list of 3–6 short bullet phrases (strings) capturing the drivers of the score
- confidence: float in [0, 1]
Do not include any other keys. Do not wrap the JSON in code fences."""


    def process(self, data: Dict[str, Any]) -> AgentResponse:
        ticker = data.get('ticker', 'UNKNOWN')
        technicals = data.get('technicals', {})

        technical_summary = f"""
Ticker: {ticker}
Current Price: ${technicals.get('current_price', 'N/A')}
50-day MA: ${technicals.get('ma_50', 'N/A')}
200-day MA: ${technicals.get('ma_200', 'N/A')}
RSI: {technicals.get('rsi', 'N/A')}
MACD: {technicals.get('macd', 'N/A')}
Volume: {technicals.get('volume', 'N/A')} (Avg: {technicals.get('avg_volume', 'N/A')})
Support: ${technicals.get('support', 'N/A')}
Resistance: ${technicals.get('resistance', 'N/A')}
"""

        user_message = f"""Analyze the following technical data for {ticker}:

{technical_summary}

Assess technical strength and price momentum. Provide only the JSON object described above."""

        raw = self.call_llm(self.system_prompt, user_message)
        js  = strip_code_fences(raw)

        try:
            result = json.loads(js)
            score = normalize_score(to_float(result.get('technical_score', 0), 0.0))
            analysis = result.get('analysis', raw)
            key_factors = result.get('key_factors', [])
            confidence = normalize_conf(result.get('confidence', 0.7))
        except json.JSONDecodeError:
            score = 0.0
            analysis = raw
            key_factors = ["Unable to parse structured response"]
            confidence = 0.6

        return AgentResponse(
            agent_name=self.agent_name,
            analysis=analysis,
            score=float(score),
            confidence=float(confidence),
            key_factors=key_factors,
            timestamp=datetime.now().isoformat()
        )

#===== Forth approach ======





class RiskAssessmentAgent(BaseAgent):
    """Assesses investment risk and portfolio fit"""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("Risk Assessment Agent", model)
        self.system_prompt = """You are a risk management analyst specializing in portfolio risk assessment.
Analyze risk metrics and provide:
1. Risk level score (0 to 1, where 0 is very low risk, 1 is very high risk)
2. Key risk factors
3. Portfolio diversification implications
4. Risk-adjusted return assessment

Return response in JSON format with keys: risk_score, analysis, key_factors, confidence"""

    def process(self, data: Dict[str, Any]) -> AgentResponse:
        ticker = data.get('ticker', 'UNKNOWN')
        risk_data = data.get('risk_metrics', {})

        risk_summary = f"""
Ticker: {ticker}
Beta: {risk_data.get('beta', 'N/A')}
Volatility (30-day): {risk_data.get('volatility', 'N/A')}%
Value at Risk (5%): ${risk_data.get('var_5', 'N/A')}
Sharpe Ratio: {risk_data.get('sharpe_ratio', 'N/A')}
Max Drawdown: {risk_data.get('max_drawdown', 'N/A')}%
Sector Correlation: {risk_data.get('sector_correlation', 'N/A')}
P/E Ratio: {risk_data.get('pe_ratio', 'N/A')}
"""

        user_message = f"""Analyze the following risk metrics for {ticker}:

{risk_summary}

Assess overall investment risk and portfolio implications."""

        raw = self.call_llm(self.system_prompt, user_message)
        js  = strip_code_fences(raw)

        try:
            result = json.loads(js)
            # Keep her 0..1 semantics but normalize to 0..1 range & clamp
            risk01 = to_float(result.get('risk_score', 0.5), 0.5)
            if risk01 > 1.0 and risk01 <= 100.0:
                risk01 = risk01 / 100.0
            elif risk01 > 1.0 and risk01 <= 10.0:
                risk01 = risk01 / 10.0
            risk01 = clamp(risk01, 0.0, 1.0)

            score = risk01  # (her design keeps risk as 0..1; no sign flip here)
            analysis = result.get('analysis', raw)
            key_factors = result.get('key_factors', [])
            confidence = normalize_conf(result.get('confidence', 0.8))
        except json.JSONDecodeError:
            score = 0.5
            analysis = raw
            key_factors = ["Unable to parse structured response"]
            confidence = 0.6

        return AgentResponse(
            agent_name=self.agent_name,
            analysis=analysis,
            score=float(score),
            confidence=float(confidence),
            key_factors=key_factors,
            timestamp=datetime.now().isoformat()
        )




class SynthesisAgent(BaseAgent):
    """Combines insights from all agents into final recommendation"""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("Research Synthesis Agent", model)
        self.system_prompt = """You are a senior investment analyst who synthesizes multiple analyses into actionable recommendations.
Given analyses from news, earnings, technical, and risk agents, provide:
1. Overall investment recommendation (STRONG BUY, BUY, HOLD, SELL, STRONG SELL)
2. Confidence level (0 to 1)
3. Key reasoning
4. Risk considerations
5. Target price range (if applicable)

Return response in JSON format with keys: recommendation, confidence, analysis, key_points, risks"""

    def process(self, agent_responses: List[AgentResponse]) -> AgentResponse:
        analyses_summary = "\n\n".join([
            f"{resp.agent_name}:\n"
            f"Score: {resp.score}\n"
            f"Analysis: {resp.analysis}\n"
            f"Key Factors: {', '.join(resp.key_factors)}"
            for resp in agent_responses
        ])

        user_message = f"""Synthesize the following analyses into a final investment recommendation:

{analyses_summary}

Provide comprehensive investment recommendation with supporting reasoning."""

        raw = self.call_llm(self.system_prompt, user_message)
        js  = strip_code_fences(raw)

        try:
            result = json.loads(js)
            recommendation = str(result.get('recommendation', 'HOLD')).upper()
            analysis = result.get('analysis', raw)
            key_factors = result.get('key_points', [])  # keep her field
            confidence = normalize_conf(result.get('confidence', 0.7))

            rec_to_score = {
                'STRONG BUY': 1.0,
                'BUY': 0.6,
                'HOLD': 0.0,
                'SELL': -0.6,
                'STRONG SELL': -1.0
            }
            score = rec_to_score.get(recommendation, 0.0)

        except json.JSONDecodeError:
            score = 0.0
            analysis = raw
            key_factors = ["Unable to parse structured response"]
            confidence = 0.6

        return AgentResponse(
            agent_name=self.agent_name,
            analysis=analysis,
            score=float(score),
            confidence=float(confidence),
            key_factors=key_factors,
            timestamp=datetime.now().isoformat()
        )




class CritiqueAgent(BaseAgent):
    """Reviews and validates analysis quality"""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("Critique & Validation Agent", model)
        self.system_prompt = """You are critique analyst who reviews investment recommendations for biases, logical errors, and completeness.
Review the synthesis and identify:
1. Logical inconsistencies
2. Potential biases
3. Missing considerations
4. Data quality issues
5. Confidence adjustment recommendation

Return response in JSON format with keys: quality_score, issues_found, suggestions, adjusted_confidence"""

    def process(self, synthesis_response: AgentResponse) -> AgentResponse:
        user_message = f"""Review this investment analysis for quality and completeness:

Recommendation: {synthesis_response.analysis}
Confidence: {synthesis_response.confidence}
Key Factors: {', '.join(synthesis_response.key_factors)}

Identify any issues, biases, or missing elements."""

        raw = self.call_llm(self.system_prompt, user_message)
        js  = strip_code_fences(raw)

        try:
            result = json.loads(js)
            quality_score = to_float(result.get('quality_score', 0.7), 0.7)
            # normalize 0..10 or 0..100 to 0..1 (display-style)
            if quality_score > 1.0 and quality_score <= 10.0:
                quality_score = quality_score / 10.0
            elif quality_score > 10.0 and quality_score <= 100.0:
                quality_score = quality_score / 100.0
            quality_score = clamp(quality_score, 0.0, 1.0)

            issues = result.get('issues_found', [])
            suggestions = result.get('suggestions', [])
            adjusted_confidence = normalize_conf(result.get('adjusted_confidence', synthesis_response.confidence))

            analysis = f"Quality Score: {quality_score}\n"
            if issues:
                analysis += f"Issues Found: {', '.join(issues)}\n"
            if suggestions:
                analysis += f"Suggestions: {', '.join(suggestions)}"

            key_factors = issues if issues else ["No major issues found"]

        except json.JSONDecodeError:
            quality_score = 0.7
            analysis = raw
            adjusted_confidence = synthesis_response.confidence
            key_factors = ["No major issues found"]

        return AgentResponse(
            agent_name=self.agent_name,
            analysis=analysis,
            score=float(quality_score),
            confidence=float(adjusted_confidence),
            key_factors=key_factors,
            timestamp=datetime.now().isoformat()
        )



