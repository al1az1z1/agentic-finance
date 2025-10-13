from __future__ import annotations
import os, json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List



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
        try:
            j = json.loads(out)
        except json.JSONDecodeError:
            j = {"sentiment_score": 0, "analysis": out, "key_factors": ["parse_fail"], "confidence": 0.5}
        return AgentResponse("News Analysis Agent", j.get("analysis",""), float(j.get("sentiment_score",0)),
                             float(j.get("confidence",0.7)), j.get("key_factors",[]), datetime.now().isoformat())

class MarketSignalsAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("Market Signals Agent", model)
    sys = ("You are a technical analyst. Return JSON with keys: "
           "technical_score, analysis, key_factors, confidence.")
    def process(self, ticker: str, technicals: Dict[str, Any]) -> AgentResponse:
        msg = f"Ticker {ticker}. Technicals: {technicals}"
        out = self.call_llm(self.sys, msg)
        try:
            j = json.loads(out)
        except json.JSONDecodeError:
            j = {"technical_score": 0, "analysis": out, "key_factors": ["parse_fail"], "confidence": 0.5}
        return AgentResponse("Market Signals Agent", j.get("analysis",""), float(j.get("technical_score",0)),
                             float(j.get("confidence",0.7)), j.get("key_factors",[]), datetime.now().isoformat())

class RiskAssessmentAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("Risk Assessment Agent", model)
    sys = ("You are a risk analyst. Return JSON with keys: "
           "risk_score, analysis, key_factors, confidence.")
    def process(self, ticker: str, risk: Dict[str, Any]) -> AgentResponse:
        msg = f"Ticker {ticker}. Risk metrics: {risk}"
        out = self.call_llm(self.sys, msg)
        try:
            j = json.loads(out)
        except json.JSONDecodeError:
            j = {"risk_score": 0.5, "analysis": out, "key_factors": ["parse_fail"], "confidence": 0.6}
        # Convert risk_score (0–1 high risk) into a signed score (negative = risky)
        score = -(float(j.get("risk_score", 0.5)) - 0.5) * 2.0
        return AgentResponse("Risk Assessment Agent", j.get("analysis",""), score,
                             float(j.get("confidence",0.7)), j.get("key_factors",[]), datetime.now().isoformat())

class SynthesisAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("Research Synthesis Agent", model)
    sys = ("You are a senior investment analyst. Return JSON with keys: "
           "recommendation, confidence, analysis, key_points, risks.")
    def process(self, parts: List[AgentResponse]) -> AgentResponse:
        blob = "\n\n".join([f"{p.agent_name} | score={p.score} | {p.analysis}" for p in parts])
        out = self.call_llm(self.sys, f"Synthesize:\n{blob}")
        try:
            j = json.loads(out)
            rec2score = {"STRONG BUY": 1.0, "BUY": 0.6, "HOLD": 0.0, "SELL": -0.6, "STRONG SELL": -1.0}
            score = rec2score.get(j.get("recommendation","HOLD").upper(), 0.0)
            kf = j.get("key_points",[]) + j.get("risks",[])
            return AgentResponse("Research Synthesis Agent", j.get("analysis",""), float(score),
                                 float(j.get("confidence",0.7)), kf, datetime.now().isoformat())
        except json.JSONDecodeError:
            return AgentResponse("Research Synthesis Agent", out, 0.0, 0.6, ["parse_fail"], datetime.now().isoformat())

class CritiqueAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__("Critique & Validation Agent", model)
    sys = ("You are a critique analyst. Return JSON with keys: "
           "quality_score, issues_found, suggestions, adjusted_confidence.")
    def process(self, synthesis: AgentResponse) -> AgentResponse:
        out = self.call_llm(self.sys, f"Review: {synthesis.analysis}")
        try:
            j = json.loads(out)
            text = f"Quality={j.get('quality_score',0.7)} | Issues={j.get('issues_found',[])} | Suggs={j.get('suggestions',[])}"
            return AgentResponse("Critique & Validation Agent", text, float(j.get("quality_score",0.7)),
                                 float(j.get("adjusted_confidence", synthesis.confidence)),
                                 j.get("issues_found",[]), datetime.now().isoformat())
        except json.JSONDecodeError:
            return AgentResponse("Critique & Validation Agent", out, 0.7, synthesis.confidence, ["parse_fail"], datetime.now().isoformat())
