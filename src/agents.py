from __future__ import annotations
import os, json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

# Import shared helpers from analysis.text
from .analysis.text import (
    strip_code_fences,
    to_float,
    clamp,
    normalize_score,
    normalize_conf,
)

# -----------------------------------------------------------------------------
# OpenAI client (safe stub for local/dev)
# -----------------------------------------------------------------------------
# Use the standard env var name
api_key = os.environ.get("OPENAI_API_KEY")

# Optional: print a very short prefix to help you debug locally
if api_key:
    print(f"OPENAI_API_KEY found: {api_key[:6]}***")
else:
    print("OPENAI_API_KEY NOT found! (running in MOCK mode)")
    # Don't set a fake key here; just run in mock.

# Initialize client if possible; otherwise fall back to mock
_client = None
try:
    # If you want to use the newer SDK:
    # from openai import OpenAI
    # _client = OpenAI()
    #
    # Or (legacy) openai.ChatCompletion API — but we'll stick to the new client interface:
    from openai import OpenAI
    if api_key:
        _client = OpenAI()
except Exception:
    _client = None


# -----------------------------------------------------------------------------
# Shared response container
# -----------------------------------------------------------------------------
@dataclass
class AgentResponse:
    agent_name: str
    analysis: str
    score: float
    confidence: float
    key_factors: List[str]
    timestamp: str


# -----------------------------------------------------------------------------
# BaseAgent
# -----------------------------------------------------------------------------
class BaseAgent:
    def __init__(self, agent_name: str, model: str = "gpt-4o"):
        self.agent_name = agent_name
        self.model = model

    def call_llm(self, system_prompt: str, user_message: str) -> str:
        # Mock path (no API key / no client)
        if _client is None:
            return json.dumps({
                "analysis": f"MOCK: {self.agent_name} processed.",
                "score": 0.0,
                "key_factors": ["mock"],
                "confidence": 0.7
            })
        try:
            resp = _client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            return resp.choices[0].message.content
        except Exception as e:
            return json.dumps({
                "analysis": f"Error: {e}",
                "score": 0.0,
                "key_factors": ["error"],
                "confidence": 0.3
            })


# -----------------------------------------------------------------------------
# News
# -----------------------------------------------------------------------------
class NewsAnalysisAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o"):
        super().__init__("News Analysis Agent", model)
        # IMPORTANT: keep everything inside one triple-quoted string
        self.system_prompt = """You are a senior financial analyst with 15+ years of experience in equity research.

Analyze the provided news articles with focus on:
1. SENTIMENT: Quantify market sentiment from -1 (very negative) to +1 (very positive)
2. MATERIALITY: How much will this impact stock price? (high/medium/low)
3. CATALYSTS: Identify specific events that could move the stock
4. RISKS: Note any red flags or concerns mentioned

SCORING GUIDELINES:
+0.8 to +1.0: Major positive catalyst (earnings beat, breakthrough product, strategic win)
+0.4 to +0.7: Positive news (growth signals, analyst upgrades, market share gains)
-0.3 to +0.3: Neutral or mixed signals
-0.7 to -0.4: Negative news (missed targets, regulatory issues, competitive threats)
-1.0 to -0.8: Major negative catalyst (fraud, bankruptcy risk, losing key customers)

IMPORTANT: 
- Use actual numbers from articles (revenue, EPS, growth rates)
- Compare to analyst expectations when mentioned
- Note if news is company-specific vs industry-wide
- Higher confidence when multiple sources agree

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

Provide sentiment analysis and impact assessment. Return only the JSON."""
        raw = self.call_llm(self.system_prompt, user_message)
        js = strip_code_fences(raw)

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


# ------------------------------------------------------------------------------
# Earnings  (COMPLETED)
# ------------------------------------------------------------------------------
class EarningsAnalysisAgent(BaseAgent):
    """Analyzes earnings reports and patterns (EPS actual vs estimate, surprise history)."""

    def __init__(self, model: str = "gpt-4o"):
        super().__init__("Earnings Analysis Agent", model)
        self.system_prompt = """You are a financial analyst specializing in earnings and fundamental analysis.

INSTRUCTIONS:
1. Analyze the earnings series objectively (EPS actual vs. estimates, surprises).
2. Identify recent beats/misses, average surprise, and beat ratio.
3. Provide a fundamental strength score from -1 (very weak) to +1 (very strong).
4. List concise key factors that justify the score.
5. Be specific with numbers when available.

EXPECTED JSON SCHEMA:
{
  "fundamental_score": float,   // -1..+1
  "analysis": string,
  "key_factors": [string],
  "confidence": float           // 0..1
}

SCORING HINTS:
- Strong positive if repeated beats, positive average surprise, improving trend.
- Negative if repeated misses, negative average surprise, deteriorating margins (if provided).
- Neutral if mixed or sparse data.

Return ONLY valid JSON with keys: fundamental_score, analysis, key_factors, confidence"""

    def process(self, data: Dict[str, Any]) -> AgentResponse:
        ticker = data.get("ticker", "UNKNOWN")
        rows = data.get("earnings", []) or []

        # Compact tabular summary to feed the model (top 8 most recent already supplied upstream)
        def row_line(r: Dict[str, Any]) -> str:
            return (
                f"- {r.get('date','?')}: estimate={r.get('EPS Estimate','n/a')}, "
                f"reported={r.get('Reported EPS','n/a')}, surprise%={r.get('Surprise(%)','n/a')}"
            )
        table = "\n".join(row_line(r) for r in rows[:12])

        user_message = f"""Company: {ticker}

Recent quarterly earnings (most recent first):
{table}

Analyze this history and return only the JSON object described in the schema."""
        raw = self.call_llm(self.system_prompt, user_message)
        js = strip_code_fences(raw)

        try:
            result = json.loads(js)
            score = normalize_score(to_float(result.get("fundamental_score", 0.0), 0.0))
            analysis = result.get("analysis", raw)
            key_factors = result.get("key_factors", [])
            confidence = normalize_conf(result.get("confidence", 0.7))
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



# -----------------------------------------------------------------------------
# Technicals
# -----------------------------------------------------------------------------
class MarketSignalsAgent(BaseAgent):
    """Performs technical analysis on market data"""

    def __init__(self, model: str = "gpt-4o"):
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

Assess technical strength and price momentum. Return only the JSON described above."""
        raw = self.call_llm(self.system_prompt, user_message)
        js = strip_code_fences(raw)

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

# -----------------------------------------------------------------------------
# Risk  (COMPLETED)
# -----------------------------------------------------------------------------
class RiskAssessmentAgent(BaseAgent):
    """Assesses investment risk and portfolio fit"""

    def __init__(self, model: str = "gpt-4o"):
        super().__init__("Risk Assessment Agent", model)
        self.system_prompt = """You are a risk management analyst specializing in portfolio risk assessment.

INSTRUCTIONS:
1. Analyze risk metrics objectively.
2. Provide a risk level score from 0 (very low risk) to 1 (very high risk).
3. Identify key risk drivers (beta, volatility, VaR, Sharpe, max drawdown, concentration/correlation).
4. Explain portfolio implications and any risk mitigants.

EXPECTED JSON SCHEMA:
{
  "risk_score": float,      // 0..1
  "analysis": string,
  "key_factors": [string],
  "confidence": float       // 0..1
}

GUIDANCE:
- Higher beta/volatility/drawdown/VaR => higher risk_score.
- Higher Sharpe => lowers effective risk_score (risk-adjusted).
- Lack of data => moderate confidence; be explicit.

Return ONLY valid JSON with keys: risk_score, analysis, key_factors, confidence"""

    def process(self, data: Dict[str, Any]) -> AgentResponse:
        ticker = data.get('ticker', 'UNKNOWN')
        risk_data = data.get('risk_metrics', {}) or {}

        # Build a compact, explicit summary. We pass both short-term and full stats if provided.
        risk_summary = f"""
Ticker: {ticker}
Beta: {risk_data.get('beta', 'N/A')}
Volatility (30-day): {risk_data.get('volatility', 'N/A')}%
Sharpe Ratio: {risk_data.get('sharpe_ratio', 'N/A')}
Max Drawdown (%): {risk_data.get('max_drawdown', 'N/A')}
Value at Risk (5% daily return): {risk_data.get('var_5', 'N/A')}
Sector Correlation: {risk_data.get('sector_correlation', 'N/A')}
P/E Ratio: {risk_data.get('pe_ratio', 'N/A')}

# Extended (may be None):
Avg Daily Return: {risk_data.get('avg_daily_return', 'N/A')}
Volatility (full window): {risk_data.get('volatility_full', 'N/A')}
"""

        user_message = f"""Analyze the following risk metrics and return only the JSON per schema:

{risk_summary}

Give a 0..1 risk_score, analysis, key_factors (bullet-style phrases), and confidence."""
        raw = self.call_llm(self.system_prompt, user_message)
        js = strip_code_fences(raw)

        try:
            result = json.loads(js)

            # Keep 0..1 semantics but normalize/clamp
            risk01 = to_float(result.get('risk_score', 0.5), 0.5)
            if 1.0 < risk01 <= 100.0:
                risk01 = risk01 / 100.0
            elif 1.0 < risk01 <= 10.0:
                risk01 = risk01 / 10.0
            risk01 = clamp(risk01, 0.0, 1.0)

            score = risk01
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


# -----------------------------------------------------------------------------
# Synthesis
# -----------------------------------------------------------------------------
class SynthesisAgent(BaseAgent):
    """Combines insights from all agents into final recommendation"""

    def __init__(self, model: str = "gpt-4o"):
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

Provide a comprehensive investment recommendation with supporting reasoning. Return only the JSON."""
        raw = self.call_llm(self.system_prompt, user_message)
        js = strip_code_fences(raw)

        try:
            result = json.loads(js)
            recommendation = str(result.get('recommendation', 'HOLD')).upper()
            analysis = result.get('analysis', raw)
            key_factors = result.get('key_points', [])
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


# -----------------------------------------------------------------------------
# Critique
# -----------------------------------------------------------------------------
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

Identify any issues, biases, or missing elements. Return only the JSON."""
        raw = self.call_llm(self.system_prompt, user_message)
        js = strip_code_fences(raw)

        try:
            result = json.loads(js)
            quality_score = to_float(result.get('quality_score', 0.7), 0.7)
            # normalize 0..10 or 0..100 to 0..1 (display-style)
            if 1.0 < quality_score <= 10.0:
                quality_score = quality_score / 10.0
            elif 10.0 < quality_score <= 100.0:
                quality_score = quality_score / 100.0
            quality_score = clamp(quality_score, 0.0, 1.0)

            issues = result.get('issues_found', [])
            suggestions = result.get('suggestions', [])
            adjusted_confidence = normalize_conf(
                result.get('adjusted_confidence', synthesis_response.confidence)
            )

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
