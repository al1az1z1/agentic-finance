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

class NewsAnalysisAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("News Analysis Agent", model)
    sys = ("You are a financial news analyst. Return JSON with keys: "
           "sentiment_score, analysis, key_factors, confidence.")
    def process(self, ticker: str, bullets: List[str]) -> AgentResponse:
        msg = f"Ticker {ticker}. News bullets:\n" + "\n".join(bullets[:5])
       
        out = self.call_llm(self.sys, msg)
        js = strip_code_fences(out)
        try:
            j = json.loads(js)
        except json.JSONDecodeError:
            j = {"sentiment_score": 0, "analysis": out, "key_factors": ["parse_fail"], "confidence": 0.5}

        sent = to_float(j.get("sentiment_score", 0.0), 0.0)
        score = normalize_score(sent)  # to [-1, 1]
        conf  = normalize_conf(j.get("confidence", 0.7))
        kf    = j.get("key_factors", [])
        return AgentResponse("News Analysis Agent", j.get("analysis",""), score, conf, kf, datetime.now().isoformat())


class MarketSignalsAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("Market Signals Agent", model)
    sys = ("You are a technical analyst. Return JSON with keys: "
           "technical_score, analysis, key_factors, confidence.")
    def process(self, ticker: str, technicals: Dict[str, Any]) -> AgentResponse:
        msg = f"Ticker {ticker}. Technicals: {technicals}"

        out = self.call_llm(self.sys, msg)
        js = strip_code_fences(out)
        try:
            j = json.loads(js)
        except json.JSONDecodeError:
            j = {"technical_score": 0, "analysis": out, "key_factors": ["parse_fail"], "confidence": 0.5}

        tech = to_float(j.get("technical_score", 0.0), 0.0)
        score = normalize_score(tech)  # to [-1, 1]
        conf  = normalize_conf(j.get("confidence", 0.7))
        kf    = j.get("key_factors", [])
        return AgentResponse("Market Signals Agent", j.get("analysis",""), score, conf, kf, datetime.now().isoformat())


class RiskAssessmentAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("Risk Assessment Agent", model)
    sys = ("You are a risk analyst. Return JSON with keys: "
           "risk_score, analysis, key_factors, confidence.")
    def process(self, ticker: str, risk: Dict[str, Any]) -> AgentResponse:
        msg = f"Ticker {ticker}. Risk metrics: {risk}"

        out = self.call_llm(self.sys, msg)
        js = strip_code_fences(out)
        try:
            j = json.loads(js)
        except json.JSONDecodeError:
            j = {"risk_score": 0.5, "analysis": out, "key_factors": ["parse_fail"], "confidence": 0.6}

        risk01 = to_float(j.get("risk_score", 0.5), 0.5)
        # normalize risk to 0..1 if given as 0..100 or 0..10
        if risk01 > 1.0 and risk01 <= 100.0:
            risk01 = risk01 / 100.0
        elif risk01 > 1.0 and risk01 <= 10.0:
            risk01 = risk01 / 10.0
        risk01 = clamp(risk01, 0.0, 1.0)

        # convert to signed score where higher risk => more negative
        score = -(risk01 - 0.5) * 2.0
        score = clamp(score, -1.0, 1.0)

        conf  = normalize_conf(j.get("confidence", 0.7))
        kf    = j.get("key_factors", [])
        return AgentResponse("Risk Assessment Agent", j.get("analysis",""), score, conf, kf, datetime.now().isoformat())


class SynthesisAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("Research Synthesis Agent", model)
    sys = ("You are a senior investment analyst. Return JSON with keys: "
           "recommendation, confidence, analysis, key_points, risks.")
    def process(self, parts: List[AgentResponse]) -> AgentResponse:
        blob = "\n\n".join([f"{p.agent_name} | score={p.score} | {p.analysis}" for p in parts])

        out = self.call_llm(self.sys, f"Synthesize:\n{blob}")
        js = strip_code_fences(out)
        try:
            j = json.loads(js)
            rec2score = {"STRONG BUY": 1.0, "BUY": 0.6, "HOLD": 0.0, "SELL": -0.6, "STRONG SELL": -1.0}
            score = rec2score.get(str(j.get("recommendation","HOLD")).upper(), 0.0)
            conf  = normalize_conf(j.get("confidence", 0.7))
            kf = j.get("key_points", []) + j.get("risks", [])
            return AgentResponse("Research Synthesis Agent", j.get("analysis",""), float(score), conf, kf, datetime.now().isoformat())
        except json.JSONDecodeError:
            return AgentResponse("Research Synthesis Agent", out, 0.0, 0.6, ["parse_fail"], datetime.now().isoformat())


class CritiqueAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("Critique & Validation Agent", model)
    sys = ("You are a critique analyst. Return JSON with keys: "
           "quality_score, issues_found, suggestions, adjusted_confidence.")
    def process(self, synthesis: AgentResponse) -> AgentResponse:

        out = self.call_llm(self.sys, f"Review: {synthesis.analysis}")
        js = strip_code_fences(out)
        try:
            j = json.loads(js)
            q  = to_float(j.get("quality_score", 0.7), 0.7)
            # quality is display-only; map 0..10 to 0..1 for score if >1
            q_score = q/10.0 if q > 1.0 and q <= 10.0 else (q/100.0 if q > 10.0 else q)
            q_score = clamp(q_score, 0.0, 1.0)
            adj = normalize_conf(j.get("adjusted_confidence", synthesis.confidence))
            text = f"Quality={j.get('quality_score',0.7)} | Issues={j.get('issues_found',[])} | Suggs={j.get('suggestions',[])}"
            return AgentResponse("Critique & Validation Agent", text, q_score, adj, j.get("issues_found",[]), datetime.now().isoformat())
        except json.JSONDecodeError:
            return AgentResponse("Critique & Validation Agent", out, 0.7, synthesis.confidence, ["parse_fail"], datetime.now().isoformat())
