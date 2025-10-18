# import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import gradio as gr
# from datetime import date, timedelta
# from src.system.orchestrator import run_pipeline

# def run(symbol, days_back, required_tags_csv):
#     start = (date.today() - timedelta(days=int(days_back))).isoformat()
#     end = date.today().isoformat()
#     tags = [t.strip() for t in required_tags_csv.split(",")] if required_tags_csv else None

#     res = run_pipeline(symbol.strip().upper(), start, end, required_tags=tags)

#     plan = "\n".join([f"• {step}" for step in res.plan])
#     agents_txt = "\n\n".join([f"[{a.agent_name}] score={a.score:.2f} conf={a.confidence:.2f}\n{a.analysis}" for a in res.agent_outputs])
#     # final_txt = f"FINAL (Synthesis)\nscore={res.final.score:.2f} conf={res.final.confidence:.2f}\n{res.final.analysis}\n\nKey: {', '.join(res.final.key_factors)}"
#     final_txt = (
#         f"FINAL (After Critique)\n"
#         f"score={res.final.score:.2f} conf={res.final.confidence:.2f}\n"
#         f"{res.final.analysis}\n\nKey: {', '.join(res.final.key_factors)}"
#     )
#     crit_txt = f"[Critique]\nscore={res.critique.score:.2f} adj_conf={res.critique.confidence:.2f}\n{res.critique.analysis}"
    

#     # Evidence tables (show first rows)
#     news_rows = res.evidence.get("top_news", [])
#     prices_rows = res.evidence.get("prices_tail", [])

#     return plan, agents_txt, final_txt, crit_txt, news_rows, prices_rows

# with gr.Blocks(title="Agentic Finance") as demo:

#     gr.Markdown("# Agentic Finance — Interactive Tester")
#     with gr.Row():
#         symbol = gr.Textbox(label="Ticker", value="AAPL")
#         days_back = gr.Slider(7, 120, value=30, step=1, label="Days Back")
#         tags = gr.Textbox(label="Required Tags (optional, comma-sep)", placeholder="earnings, product")
#     run_btn = gr.Button("Run")

#     plan = gr.Textbox(label="Plan", lines=6)
#     agents = gr.Textbox(label="Agent Outputs", lines=14)
#     final = gr.Textbox(label="Final Recommendation", lines=8)
#     crit = gr.Textbox(label="Critique", lines=6)

#     news_tbl = gr.Dataframe(headers=["published_at","source","title","summary","url","overall_sentiment","tags","numbers"], label="Top News (evidence)", wrap=True)
#     prices_tbl = gr.Dataframe(label="Recent Prices (evidence)")

#     run_btn.click(run, inputs=[symbol, days_back, tags], outputs=[plan, agents, final, crit, news_tbl, prices_tbl])

# if __name__ == "__main__":
#     demo.launch()




# Second appriach

# import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import gradio as gr
# import pandas as pd
# from datetime import date, timedelta
# from src.system.orchestrator import run_pipeline

# def run(symbol, days_back, required_tags_csv):
#     start = (date.today() - timedelta(days=int(days_back))).isoformat()
#     end = date.today().isoformat()
#     tags = [t.strip() for t in required_tags_csv.split(",")] if required_tags_csv else None

#     res = run_pipeline(symbol.strip().upper(), start, end, required_tags=tags)

#     # Detect if optimizer re-synthesis ran (Initial Synthesis exists and differs from final)
#     optimizer_ran = False
#     for a in res.agent_outputs:
#         if a.agent_name == "Initial Synthesis":
#             optimizer_ran = (a.analysis.strip() != res.final.analysis.strip())
#             break

#     plan = "\n".join([f"• {step}" for step in res.plan])

#     agents_txt = "\n\n".join([
#         f"[{a.agent_name}] score={a.score:.2f} conf={a.confidence:.2f}\n{a.analysis}"
#         for a in res.agent_outputs
#     ])

#     # Build final block with conditional headline
#     headline = "FINAL (After Critique)" if optimizer_ran else "FINAL (Synthesis)"
#     final_txt = (
#         f"{headline}\n"
#         f"score={res.final.score:.2f} conf={res.final.confidence:.2f}\n"
#         f"{res.final.analysis}\n\nKey: {', '.join(res.final.key_factors)}"
#     )

#     crit_txt = (
#         f"[Critique]\n"
#         f"score={res.critique.score:.2f} adj_conf={res.critique.confidence:.2f}\n"
#         f"{res.critique.analysis}"
#     )

#     # Evidence tables (show first rows)
#     news_rows = res.evidence.get("top_news", [])
#     prices_rows = res.evidence.get("prices_tail", [])

#     # Optional: friendly hint when no news evidence
#     if isinstance(news_rows, pd.DataFrame) and news_rows.empty:
#         agents_txt += "\n\n[Note] No news items matched filters or API limits were hit today."

#     return plan, agents_txt, final_txt, crit_txt, news_rows, prices_rows

# with gr.Blocks(title="Agentic Finance") as demo:
#     gr.Markdown("# Agentic Finance — Interactive Tester")
#     with gr.Row():
#         symbol = gr.Textbox(label="Ticker", value="AAPL")
#         days_back = gr.Slider(7, 120, value=30, step=1, label="Days Back")
#         tags = gr.Textbox(label="Required Tags (optional, comma-sep)", placeholder="earnings, product")
#     run_btn = gr.Button("Run")

#     plan = gr.Textbox(label="Plan", lines=6)
#     agents = gr.Textbox(label="Agent Outputs", lines=14)
#     final = gr.Textbox(label="Final Recommendation", lines=8)
#     crit = gr.Textbox(label="Critique", lines=6)

#     news_tbl = gr.Dataframe(
#         headers=["published_at","source","title","summary","url","overall_sentiment","tags","numbers"],
#         label="Top News (evidence)",
#         wrap=True
#     )
#     prices_tbl = gr.Dataframe(label="Recent Prices (evidence)")

#     run_btn.click(run, inputs=[symbol, days_back, tags], outputs=[plan, agents, final, crit, news_tbl, prices_tbl])

# if __name__ == "__main__":
#     demo.launch()


# third approach

# import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import gradio as gr
# import pandas as pd
# from datetime import date, timedelta
# from src.system.orchestrator import run_pipeline

# # def run(symbol, days_back, required_tags_csv):
# #     start = (date.today() - timedelta(days=int(days_back))).isoformat()
# #     end = date.today().isoformat()
# #     tags = [t.strip() for t in required_tags_csv.split(",")] if required_tags_csv else None

# #     res = run_pipeline(symbol.strip().upper(), start, end, required_tags=tags)

# #     # Detect if optimizer re-synthesis ran (Initial Synthesis exists and differs from final)
# #     optimizer_ran = False
# #     for a in res.agent_outputs:
# #         if a.agent_name == "Initial Synthesis":
# #             optimizer_ran = (a.analysis.strip() != res.final.analysis.strip())
# #             break

# #     plan = "\n".join([f"• {step}" for step in res.plan])

# #     agents_txt = "\n\n".join([
# #         f"[{a.agent_name}] score={a.score:.2f} conf={a.confidence:.2f}\n{a.analysis}"
# #         for a in res.agent_outputs
# #     ])

# #     # Evidence tables (show first rows)
# #     news_rows = res.evidence.get("top_news", [])
# #     prices_rows = res.evidence.get("prices_tail", [])

# #     # Helpful note when news evidence is empty
# #     if isinstance(news_rows, pd.DataFrame) and news_rows.empty:
# #         agents_txt += "\n\n[Note] No news items matched filters or API limits were hit today."

# #     # ---- Critique FIRST (so it appears above Final in the UI) ----
# #     crit_txt = (
# #         f"[Critique]\n"
# #         f"score={res.critique.score:.2f} adj_conf={res.critique.confidence:.2f}\n"
# #         f"{res.critique.analysis}"
# #     )

# #     # ---- Final AFTER Critique ----
# #     # Always label as After Critique; orchestrator already sets res.final to the post-critique synthesis.
# #     headline = "FINAL (After Critique)"
# #     opt_line = "[Optimizer ran: YES]" if optimizer_ran else "[Optimizer ran: NO]"
# #     final_txt = (
# #         f"{headline}\n{opt_line}\n"
# #         f"score={res.final.score:.2f} conf={res.final.confidence:.2f}\n"
# #         f"{res.final.analysis}\n\nKey: {', '.join(res.final.key_factors)}"
# #     )

# #     # IMPORTANT: return order matches component outputs order (crit BEFORE final)
# #     return plan, agents_txt, crit_txt, final_txt, news_rows, prices_rows



# def run(symbol, days_back, required_tags_csv):
#     import json
#     import pandas as pd

#     def as_text(x):
#         if x is None:
#             return ""
#         if isinstance(x, str):
#             return x
#         if isinstance(x, dict) or isinstance(x, list):
#             # pretty JSON for readability / stable comparisons
#             return json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True)
#         return str(x)

#     def clean(s: str) -> str:
#         # normalize whitespace and strip code fences if any slipped through
#         if not isinstance(s, str):
#             s = as_text(s)
#         s = s.strip()
#         if s.startswith("```"):
#             s = s.strip("`").strip()
#         return s

#     def to_df(x):
#         if isinstance(x, pd.DataFrame):
#             return x
#         if x is None:
#             return pd.DataFrame()
#         try:
#             return pd.DataFrame(x)
#         except Exception:
#             return pd.DataFrame()

#     start = (date.today() - timedelta(days=int(days_back))).isoformat()
#     end = date.today().isoformat()
#     tags = [t.strip() for t in required_tags_csv.split(",")] if required_tags_csv else None

#     res = run_pipeline(symbol.strip().upper(), start, end, required_tags=tags)

#     # Detect if optimizer re-synthesis ran (compare Initial vs Final, after normalizing to text)
#     optimizer_ran = False
#     init = next((a for a in res.agent_outputs if a.agent_name in {"Initial Synthesis", "Research Synthesis Agent", "SynthesisAgent"}), None)
#     if init is not None:
#         init_txt = clean(as_text(init.analysis))
#         final_txt_norm = clean(as_text(res.final.analysis))
#         optimizer_ran = (init_txt != final_txt_norm)

#     plan = "\n".join([f"• {step}" for step in res.plan])

#     agents_txt = "\n\n".join([
#         f"[{a.agent_name}] score={a.score:.2f} conf={a.confidence:.2f}\n{as_text(a.analysis)}"
#         for a in res.agent_outputs
#     ])

#     # Evidence tables (force DataFrame)
#     news_rows = to_df(res.evidence.get("top_news", []))
#     prices_rows = to_df(res.evidence.get("prices_tail", []))

#     # Helpful note when news evidence is empty
#     if news_rows.empty:
#         agents_txt += "\n\n[Note] No news items matched filters or API limits were hit today."

#     # ---- Critique FIRST (so it appears above Final in the UI) ----
#     crit_txt = (
#         f"[Critique]\n"
#         f"score={res.critique.score:.2f} adj_conf={res.critique.confidence:.2f}\n"
#         f"{as_text(res.critique.analysis)}"
#     )

#     # ---- Final AFTER Critique ----
#     headline = "FINAL (After Critique)"
#     opt_line = "[Optimizer ran: YES]" if optimizer_ran else "[Optimizer ran: NO]"
#     final_txt = (
#         f"{headline}\n{opt_line}\n"
#         f"score={res.final.score:.2f} conf={res.final.confidence:.2f}\n"
#         f"{as_text(res.final.analysis)}\n\nKey: {', '.join(res.final.key_factors)}"
#     )

#     return plan, agents_txt, crit_txt, final_txt, news_rows, prices_rows


# with gr.Blocks(title="Agentic Finance") as demo:
#     gr.Markdown("# Agentic Finance — Interactive Tester")

#     with gr.Row():
#         symbol = gr.Textbox(label="Ticker", value="AAPL")
#         days_back = gr.Slider(7, 120, value=30, step=1, label="Days Back")
#         tags = gr.Textbox(label="Required Tags (optional, comma-sep)", placeholder="earnings, product")
#     run_btn = gr.Button("Run")

#     plan = gr.Textbox(label="Plan", lines=6)
#     agents = gr.Textbox(label="Agent Outputs", lines=14)

#     #  Critique ABOVE Final
#     crit = gr.Textbox(label="Critique", lines=8)
#     final = gr.Textbox(label="Final Recommendation", lines=10)

#     news_tbl = gr.Dataframe(
#         headers=["published_at","source","title","summary","url","overall_sentiment","tags","numbers"],
#         label="Top News (evidence)",
#         wrap=True
#     )
#     prices_tbl = gr.Dataframe(label="Recent Prices (evidence)")

#     # Outputs order updated to: plan, agents, CRITIQUE, FINAL, news, prices
#     run_btn.click(
#         run,
#         inputs=[symbol, days_back, tags],
#         outputs=[plan, agents, crit, final, news_tbl, prices_tbl]
#     )

# if __name__ == "__main__":
#     demo.launch()

# Third approach



# forth approach
# ui/gradio_app.py

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import pandas as pd
from datetime import date, timedelta
from src.system.orchestrator import run_pipeline


def run(symbol, days_back, required_tags_csv):
    import json
    import pandas as pd

    # ---------- helpers ----------
    def as_text(x):
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, (dict, list)):
            # pretty JSON for readability / stable comparisons
            return json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True)
        return str(x)

    def clean(s: str) -> str:
        # normalize whitespace and strip code fences if any slipped through
        if not isinstance(s, str):
            s = as_text(s)
        s = s.strip()
        if s.startswith("```"):
            s = s.strip("`").strip()
        return s

    def synth_to_prose(obj):
        # If it's already text, just return normalized text
        if not isinstance(obj, dict):
            return clean(as_text(obj))

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
            if ma50 is not None or ma200 is not None:
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
            # keep order stable-ish
            news_bits = []
            for k in ("sentiment", "growth potential", "competitive landscape"):
                if k in news:
                    news_bits.append(f"{k}: {news[k]}")
            # include any other keys if present
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

    def to_df(x):
        if isinstance(x, pd.DataFrame):
            return x
        if x is None:
            return pd.DataFrame()
        try:
            return pd.DataFrame(x)
        except Exception:
            return pd.DataFrame()
    # ---------- /helpers ----------

    start = (date.today() - timedelta(days=int(days_back))).isoformat()
    end = date.today().isoformat()
    tags = [t.strip() for t in required_tags_csv.split(",")] if required_tags_csv else None

    res = run_pipeline(symbol.strip().upper(), start, end, required_tags=tags)

    # Detect if optimizer re-synthesis ran (compare Initial vs Final on normalized JSON text)
    optimizer_ran = False
    init = next((a for a in res.agent_outputs if a.agent_name in {"Initial Synthesis", "Research Synthesis Agent", "SynthesisAgent"}), None)
    if init is not None:
        init_txt = clean(as_text(init.analysis))
        final_txt_norm = clean(as_text(res.final.analysis))
        optimizer_ran = (init_txt != final_txt_norm)

    plan = "\n".join([f"• {step}" for step in res.plan])

    # Agents panel: Synthesis agents shown as prose; others as text
    agents_txt = "\n\n".join([
        (
            f"[{a.agent_name}] score={a.score:.2f} conf={a.confidence:.2f}\n"
            f"{synth_to_prose(a.analysis) if ('synthesis' in a.agent_name.lower()) else clean(as_text(a.analysis))}"
        )
        for a in res.agent_outputs
    ])

    # Evidence tables (force DataFrame)
    news_rows = to_df(res.evidence.get("top_news", []))
    prices_rows = to_df(res.evidence.get("prices_tail", []))

    # Helpful note when news evidence is empty
    if news_rows.empty:
        agents_txt += "\n\n[Note] No news items matched filters or API limits were hit today."

    # ---- Critique FIRST (so it appears above Final in the UI) ----
    crit_txt = (
        f"[Critique]\n"
        f"score={res.critique.score:.2f} adj_conf={res.critique.confidence:.2f}\n"
        f"{clean(as_text(res.critique.analysis))}"
    )

    # ---- Final AFTER Critique ----
    headline = "FINAL (After Critique)"
    opt_line = "[Optimizer ran: YES]" if optimizer_ran else "[Optimizer ran: NO]"
    final_txt = (
        f"{headline}\n{opt_line}\n"
        f"score={res.final.score:.2f} conf={res.final.confidence:.2f}\n"
        f"{synth_to_prose(res.final.analysis)}\n\nKey: {', '.join(res.final.key_factors)}"
    )

    # IMPORTANT: return order matches component outputs order (crit BEFORE final)
    return plan, agents_txt, crit_txt, final_txt, news_rows, prices_rows


with gr.Blocks(title="Agentic Finance") as demo:
    gr.Markdown("# Agentic Finance — Interactive Tester")

    with gr.Row():
        symbol = gr.Textbox(label="Ticker", value="AAPL")
        days_back = gr.Slider(7, 120, value=30, step=1, label="Days Back")
        tags = gr.Textbox(label="Required Tags (optional, comma-sep)", placeholder="earnings, product")
    run_btn = gr.Button("Run")

    plan = gr.Textbox(label="Plan", lines=6)
    agents = gr.Textbox(label="Agent Outputs", lines=14)

    #  Critique ABOVE Final
    crit = gr.Textbox(label="Critique", lines=8)
    final = gr.Textbox(label="Final Recommendation", lines=10)

    news_tbl = gr.Dataframe(
        headers=["published_at","source","title","summary","url","overall_sentiment","tags","numbers"],
        label="Top News (evidence)",
        wrap=True
    )
    prices_tbl = gr.Dataframe(label="Recent Prices (evidence)")

    # Outputs order: plan, agents, CRITIQUE, FINAL, news, prices
    run_btn.click(
        run,
        inputs=[symbol, days_back, tags],
        outputs=[plan, agents, crit, final, news_tbl, prices_tbl]
    )

if __name__ == "__main__":
    demo.launch()
