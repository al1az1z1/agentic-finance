import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from datetime import date, timedelta
from src.system.orchestrator import run_pipeline

def run(symbol, days_back, required_tags_csv):
    start = (date.today() - timedelta(days=int(days_back))).isoformat()
    end = date.today().isoformat()
    tags = [t.strip() for t in required_tags_csv.split(",")] if required_tags_csv else None

    res = run_pipeline(symbol.strip().upper(), start, end, required_tags=tags)

    plan = "\n".join([f"• {step}" for step in res.plan])
    agents_txt = "\n\n".join([f"[{a.agent_name}] score={a.score:.2f} conf={a.confidence:.2f}\n{a.analysis}" for a in res.agent_outputs])
    final_txt = f"FINAL (Synthesis)\nscore={res.final.score:.2f} conf={res.final.confidence:.2f}\n{res.final.analysis}\n\nKey: {', '.join(res.final.key_factors)}"
    crit_txt = f"[Critique]\nscore={res.critique.score:.2f} adj_conf={res.critique.confidence:.2f}\n{res.critique.analysis}"

    # Evidence tables (show first rows)
    news_rows = res.evidence.get("top_news", [])
    prices_rows = res.evidence.get("prices_tail", [])

    return plan, agents_txt, final_txt, crit_txt, news_rows, prices_rows

with gr.Blocks(title="Agentic Finance") as demo:

    gr.Markdown("# Agentic Finance — Interactive Tester")
    with gr.Row():
        symbol = gr.Textbox(label="Ticker", value="AAPL")
        days_back = gr.Slider(7, 120, value=30, step=1, label="Days Back")
        tags = gr.Textbox(label="Required Tags (optional, comma-sep)", placeholder="earnings, product")
    run_btn = gr.Button("Run")

    plan = gr.Textbox(label="Plan", lines=6)
    agents = gr.Textbox(label="Agent Outputs", lines=14)
    final = gr.Textbox(label="Final Recommendation", lines=8)
    crit = gr.Textbox(label="Critique", lines=6)

    news_tbl = gr.Dataframe(headers=["published_at","source","title","summary","url","overall_sentiment","tags","numbers"], label="Top News (evidence)", wrap=True)
    prices_tbl = gr.Dataframe(label="Recent Prices (evidence)")

    run_btn.click(run, inputs=[symbol, days_back, tags], outputs=[plan, agents, final, crit, news_tbl, prices_tbl])

if __name__ == "__main__":
    demo.launch()
