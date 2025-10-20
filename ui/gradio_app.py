import os, sys, traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import pandas as pd
from datetime import date, timedelta
from src.system.orchestrator import run_pipeline

# ---------- small helpers ----------
def _truncate(s: str, max_len: int = 8000) -> str:
    if not isinstance(s, str):
        s = str(s)
    return (s[: max_len - 20] + " … (truncated)") if len(s) > max_len else s

def _as_text(x):
    import json
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True)
    return str(x)

def _clean(s: str) -> str:
    s = _as_text(s)
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
    return s

def _synth_to_prose(obj):
    if not isinstance(obj, dict):
        return _clean(_as_text(obj))
    parts = []

    ms = obj.get("market_signals") or {}
    if ms:
        ms_bits = []
        cp = ms.get("current_price")
        if isinstance(cp, (int, float)):
            ms_bits.append(f"price ${cp:,.2f}")
        ma = ms.get("moving_averages") or {}
        ma50 = ma.get("50_day")
        ma200 = ma.get("200_day")
        if (ma50 is not None) or (ma200 is not None):
            ms_bits.append(f"vs 50D {ma50}, 200D {ma200}")
        rsi = ms.get("RSI")
        if rsi is not None:
            ms_bits.append(f"RSI {rsi}")
        trend = ms.get("trend")
        if trend:
            ms_bits.append(trend)
        vol = ms.get("volume") or {}
        vcur, vavg = vol.get("current"), vol.get("average")
        if vcur is not None and vavg is not None:
            ms_bits.append(f"volume {vcur:,} vs avg {vavg:,}")
        if ms_bits:
            parts.append("Technicals: " + ", ".join(str(x) for x in ms_bits if x))

    news = obj.get("news") or {}
    if news:
        news_bits = []
        for k in ("sentiment", "growth potential", "competitive landscape"):
            if k in news:
                news_bits.append(f"{k}: {news[k]}")
        for k, v in news.items():
            if k not in ("sentiment", "growth potential", "competitive landscape"):
                news_bits.append(f"{k}: {v}")
        parts.append("News: " + "; ".join(news_bits))

    risk = obj.get("risk_assessment") or {}
    if risk:
        risk_bits = []
        for k in ("volatility", "data_gaps", "idiosyncratic_risks"):
            if k in risk:
                risk_bits.append(f"{k}: {risk[k]}")
        for k, v in risk.items():
            if k not in ("volatility", "data_gaps", "idiosyncratic_risks"):
                risk_bits.append(f"{k}: {v}")
        parts.append("Risk: " + "; ".join(risk_bits))

    return "\n".join(parts).strip()

def _to_df(x):
    if isinstance(x, pd.DataFrame):
        return x
    if x is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(x)
    except Exception:
        return pd.DataFrame()
# ---------- /helpers ----------

def run(symbol, days_back, required_tags_csv):
    try:
        start = (date.today() - timedelta(days=int(days_back))).isoformat()
        end = date.today().isoformat()
        tags = [t.strip() for t in required_tags_csv.split(",")] if required_tags_csv else None

        res = run_pipeline(symbol.strip().upper(), start, end, required_tags=tags)

        # Detect optimizer re-synthesis
        optimizer_ran = False
        init = next((a for a in res.agent_outputs if a.agent_name in {"Initial Synthesis", "Research Synthesis Agent", "SynthesisAgent"}), None)
        if init is not None:
            init_txt = _clean(_as_text(init.analysis))
            final_txt_norm = _clean(_as_text(res.final.analysis))
            optimizer_ran = (init_txt != final_txt_norm)

        plan = "\n".join([f"• {step}" for step in res.plan])

        # Agents panel (truncate to keep websocket payload small)
        agents_txt = "\n\n".join([
            (
                f"[{a.agent_name}] score={a.score:.2f} conf={a.confidence:.2f}\n"
                f"{_synth_to_prose(a.analysis) if ('synthesis' in a.agent_name.lower()) else _clean(_as_text(a.analysis))}"
            )
            for a in res.agent_outputs
        ])
        agents_txt = _truncate(agents_txt, 15000)

        # Evidence tables
        news_rows      = _to_df(res.evidence.get("top_news", []))
        prices_rows    = _to_df(res.evidence.get("prices_tail", []))
        earnings_rows  = _to_df(res.evidence.get("earnings_head", []))   # NEW
        risk_rows      = _to_df(res.evidence.get("risk_metrics", []))    # NEW (single-row DF)

        if news_rows.empty:
            agents_txt += "\n\n[Note] No news items matched filters or API limits were hit today."

        crit_txt = (
            f"[Critique]\n"
            f"score={res.critique.score:.2f} adj_conf={res.critique.confidence:.2f}\n"
            f"{_clean(_as_text(res.critique.analysis))}"
        )
        crit_txt = _truncate(crit_txt, 6000)

        headline = "FINAL (After Critique)"
        opt_line = "[Optimizer ran: YES]" if optimizer_ran else "[Optimizer ran: NO]"
        final_txt = (
            f"{headline}\n{opt_line}\n"
            f"score={res.final.score:.2f} conf={res.final.confidence:.2f}\n"
            f"{_synth_to_prose(res.final.analysis)}\n\nKey: {', '.join(res.final.key_factors)}"
        )
        final_txt = _truncate(final_txt, 8000)

        # Return order MUST match component outputs order
        return (
            plan,
            agents_txt,
            crit_txt,
            final_txt,
            news_rows,
            prices_rows,
            earnings_rows,   # NEW
            risk_rows        # NEW
        )

    except Exception:
        tb = traceback.format_exc()
        err = f"[FATAL] An exception occurred in run():\n{tb}"
        blank_df = pd.DataFrame()
        return "run() error — see Critique tab", _truncate(err, 15000), _truncate(err, 6000), _truncate(err, 8000), blank_df, blank_df, blank_df, blank_df


with gr.Blocks(title="Agentic Finance") as demo:
    gr.Markdown("# Agentic Finance — Interactive Tester")

    with gr.Row():
        symbol = gr.Textbox(label="Ticker", value="AAPL")
        days_back = gr.Slider(7, 120, value=30, step=1, label="Days Back")
        tags = gr.Textbox(label="Required Tags (optional, comma-sep)", placeholder="earnings, product")
    run_btn = gr.Button("Run")

    plan   = gr.Textbox(label="Plan", lines=6)
    agents = gr.Textbox(label="Agent Outputs", lines=14)
    crit   = gr.Textbox(label="Critique", lines=8)
    final  = gr.Textbox(label="Final Recommendation", lines=10)

    news_tbl     = gr.Dataframe(
        headers=["published_at","source","title","summary","url","overall_sentiment","tags","numbers"],
        label="Top News (evidence)",
        wrap=True
    )
    prices_tbl   = gr.Dataframe(label="Recent Prices (evidence)")
    earnings_tbl = gr.Dataframe(label="Earnings (evidence)")           # NEW
    risk_tbl     = gr.Dataframe(label="Risk Metrics (evidence)")       # NEW

    run_btn.click(
        run,
        inputs=[symbol, days_back, tags],
        outputs=[plan, agents, crit, final, news_tbl, prices_tbl, earnings_tbl, risk_tbl]  # NEW outputs
    )

if __name__ == "__main__":
    # Queue/launch shim for broad Gradio compatibility
    try:
        demo.queue()
    except TypeError:
        try:
            demo.queue(max_size=16)
        except TypeError:
            pass

    try:
        demo.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True
        )
    except TypeError:
        demo.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7860
        )
