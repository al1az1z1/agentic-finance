import os, sys, traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import pandas as pd
from datetime import date, timedelta
from src.system.orchestrator import run_pipeline

# ---------- small helpers ----------
def _truncate(s: str, max_len: int = 8000) -> str:
    """We trim long text so UI stays responsive."""
    if not isinstance(s, str):
        s = str(s)
    return (s[: max_len - 20] + " … (truncated)") if len(s) > max_len else s

def _as_text(x):
    """We return a string for any JSON-like input; pretty-print lists/dicts."""
    import json
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True)
    return str(x)

def _clean(s: str) -> str:
    """We normalize whitespace and strip accidental code fences."""
    s = _as_text(s).strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
    return s

def _synth_to_prose(obj):
    """We turn a structured synthesis dict into short readable lines for the UI."""
    if not isinstance(obj, dict):
        return _clean(_as_text(obj))
    parts = []

    # --- market signals summary ---
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

    # --- news summary ---
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

    # --- risk summary ---
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
    """We coerce anything into a DataFrame (or empty one on failure)."""
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
    """We execute the pipeline and format results for the UI panels."""
    try:
        # --- date window & input tags ---
        start = (date.today() - timedelta(days=int(days_back))).isoformat()
        end = date.today().isoformat()
        tags = [t.strip() for t in required_tags_csv.split(",")] if required_tags_csv else None

        # --- run the orchestrator (final already equals synth_v2 when optimizer runs) ---
        res = run_pipeline(symbol.strip().upper(), start, end, required_tags=tags)

        # --- detect whether optimizer (synth_v2) ran by comparing v1 vs final ---
        optimizer_ran = False
        init = next(
            (a for a in res.agent_outputs
             if a.agent_name in {"Initial Synthesis", "Research Synthesis Agent", "SynthesisAgent"}),
            None
        )
        if init is not None:
            init_txt = _clean(_as_text(init.analysis))
            final_txt_norm = _clean(_as_text(res.final.analysis))
            optimizer_ran = (init_txt != final_txt_norm)

        # --- step plan (compact bullets) ---
        plan = "\n".join([f"• {step}" for step in res.plan])

        # --- agent outputs (shortened to keep payload light) ---
        agents_txt = "\n\n".join([
            (
                f"[{a.agent_name}] score={a.score:.2f} conf={a.confidence:.2f}\n"
                f"{_synth_to_prose(a.analysis) if ('synthesis' in a.agent_name.lower()) else _clean(_as_text(a.analysis))}"
            )
            for a in res.agent_outputs
        ])
        agents_txt = _truncate(agents_txt, 15000)

        # --- evidence tables (safe to be empty) ---
        news_rows      = _to_df(res.evidence.get("top_news", []))
        prices_rows    = _to_df(res.evidence.get("prices_tail", []))
        earnings_rows  = _to_df(res.evidence.get("earnings_head", []))   # from orchestrator
        risk_rows      = _to_df(res.evidence.get("risk_metrics", []))    # single-row DF

        if news_rows.empty:
            agents_txt += "\n\n[Note] No news items matched filters or API limits were hit."

        # --- critique panel ---
        crit_txt = (
            f"[Critique]\n"
            f"score={res.critique.score:.2f} adj_conf={res.critique.confidence:.2f}\n"
            f"{_clean(_as_text(res.critique.analysis))}"
        )
        crit_txt = _truncate(crit_txt, 6000)

        # --- final panel (auto-label v2 when optimizer ran) ---
        headline = "FINAL (v2 after Critique)" if optimizer_ran else "FINAL (v1)"
        opt_line = "[Optimizer ran: YES]" if optimizer_ran else "[Optimizer ran: NO]"
        final_txt = (
            f"{headline}\n{opt_line}\n"
            f"score={res.final.score:.2f} conf={res.final.confidence:.2f}\n"
            f"{_synth_to_prose(res.final.analysis)}\n\nKey: {', '.join(res.final.key_factors)}"
        )
        final_txt = _truncate(final_txt, 8000)

        # --- return in the same order the UI wires outputs ---
        return (
            plan,
            agents_txt,
            crit_txt,
            final_txt,
            news_rows,
            prices_rows,
            earnings_rows,
            risk_rows
        )

    except Exception:
        # We catch everything so the app doesn’t crash; we show traceback in panels.
        tb = traceback.format_exc()
        err = f"[FATAL] An exception occurred in run():\n{tb}"
        blank_df = pd.DataFrame()
        return "run() error — see Critique tab", _truncate(err, 15000), _truncate(err, 6000), _truncate(err, 8000), blank_df, blank_df, blank_df, blank_df


# ---- Gradio layout ----
with gr.Blocks(title="Agentic Finance") as demo:
    gr.Markdown("# Agentic Finance — Interactive Tester")

    with gr.Row():
        symbol = gr.Textbox(label="Ticker", value="AAPL")                  # we type the ticker here
        days_back = gr.Slider(7, 120, value=30, step=1, label="Days Back") # we control the lookback window
        tags = gr.Textbox(label="Required Tags (optional, comma-sep)", placeholder="earnings, product")
    run_btn = gr.Button("Run")

    # Text panels
    plan   = gr.Textbox(label="Plan", lines=6)                  # we show the step plan
    agents = gr.Textbox(label="Agent Outputs", lines=14)        # we show all agent messages
    crit   = gr.Textbox(label="Critique", lines=8)              # we show the critique message
    final  = gr.Textbox(label="Final Recommendation", lines=10) # we show v1 or v2 based on optimizer

    # Evidence tables
    news_tbl     = gr.Dataframe(
        headers=["published_at","source","title","summary","url","overall_sentiment","tags","numbers"],
        label="Top News (evidence)",
        wrap=True
    )
    prices_tbl   = gr.Dataframe(label="Recent Prices (evidence)")
    earnings_tbl = gr.Dataframe(label="Earnings (evidence)")
    risk_tbl     = gr.Dataframe(label="Risk Metrics (evidence)")

    # Wire click → run() → panels
    run_btn.click(
        run,
        inputs=[symbol, days_back, tags],
        outputs=[plan, agents, crit, final, news_tbl, prices_tbl, earnings_tbl, risk_tbl]
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
    
    if __name__ == "__main__":
        import socket

    def _get_free_port(start=7860, end=7890):
        for p in range(start, end + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        return None  # let Gradio auto-pick if needed

    # Try to queue; ignore older Gradio signatures
    try:
        demo.queue()
    except TypeError:
        try:
            demo.queue(max_size=16)
        except TypeError:
            pass

    port = _get_free_port()  # None → let Gradio auto-choose

    try:
        demo.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=port,      # may be None; Gradio will auto-pick
            show_error=True
        )
    except OSError:
        # Fallback: force auto-pick any free port
        demo.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=None,
            show_error=True
        )
