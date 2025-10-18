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



# ====== Sunitha's version without comments ======


class NewsAnalysisAgent(BaseAgent):
    """Analyzes financial news sentiment and impact"""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("News Analysis Agent", model)
        self.system_prompt = """You are a financial news analyst specializing in sentiment analysis.

INSTRUCTIONS:
1. Analyze news articles objectively
2. Consider both positive and negative aspects
3. Provide a sentiment score from -1 (very negative) to +1 (very positive)
4. Identify key factors driving the sentiment
5. Assess potential stock price impact

EXAMPLE OUTPUT:
{
  "sentiment_score": 0.75,
  "analysis": "Strong positive sentiment driven by earnings beat and product launch",
  "key_factors": ["Earnings exceeded expectations", "New product well-received"],
  "confidence": 0.85
}

Return ONLY valid JSON with keys: sentiment_score, analysis, key_factors, confidence"""

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
        self.system_prompt = """You are a technical analyst specializing in market signals and price patterns.

INSTRUCTIONS:
1. Analyze technical indicators objectively
2. Assess technical strength from -1 (very bearish) to +1 (very bullish)
3. Identify support/resistance levels
4. Evaluate trend direction and momentum
5. Consider volume patterns

EXAMPLE OUTPUT:
{
  "technical_score": 0.65,
  "analysis": "Bullish technical setup with price above key moving averages",
  "key_factors": ["Price above 50-day MA", "RSI indicates strength", "Volume confirming uptrend"],
  "confidence": 0.75
}

Return ONLY valid JSON with keys: technical_score, analysis, key_factors, confidence"""


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

INSTRUCTIONS:
1. Analyze risk metrics objectively
2. Provide risk level score from 0 (very low risk) to 1 (very high risk)
3. Identify key risk factors
4. Assess portfolio diversification implications
5. Evaluate risk-adjusted returns

EXAMPLE OUTPUT:
{
  "risk_score": 0.45,
  "analysis": "Moderate risk profile with acceptable volatility and strong Sharpe ratio",
  "key_factors": ["Beta of 1.15 indicates moderate volatility", "Strong Sharpe ratio", "Manageable drawdown"],
  "confidence": 0.82
}

Return ONLY valid JSON with keys: risk_score, analysis, key_factors, confidence"""

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

INSTRUCTIONS:
1. Review all agent analyses objectively
2. Weigh different factors appropriately
3. Provide clear investment recommendation (STRONG BUY, BUY, HOLD, SELL, STRONG SELL)
4. State confidence level (0 to 1)
5. Summarize key reasoning
6. Note important risks

EXAMPLE OUTPUT:
{
  "recommendation": "BUY",
  "confidence": 0.78,
  "analysis": "Strong fundamentals and positive technical signals support a buy recommendation despite moderate risk",
  "key_points": ["Earnings beat expectations", "Technical breakout", "Acceptable risk profile"],
  "risks": ["Market volatility", "Sector headwinds"]
}

Return ONLY valid JSON with keys: recommendation, confidence, analysis, key_points, risks"""

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
        self.system_prompt = """You are a critique analyst who reviews investment recommendations for biases, logical errors, and completeness.

INSTRUCTIONS:
1. Review the synthesis objectively
2. Identify logical inconsistencies
3. Detect potential biases
4. Note missing considerations
5. Assess data quality
6. Recommend confidence adjustments

EXAMPLE OUTPUT:
{
  "quality_score": 0.82,
  "issues_found": ["Limited macroeconomic analysis"],
  "suggestions": ["Consider Federal Reserve policy impact", "Add sector comparison"],
  "adjusted_confidence": 0.75
}

Return ONLY valid JSON with keys: quality_score, issues_found, suggestions, adjusted_confidence"""

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



