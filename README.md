# agentic-finance
AAI-520 Final Project: Multi-Agent Financial Analysis System using Agentic AI

## Project Directory Layout

agentic-finance/
├─ data/                     # Local data storage
│  ├─ cache/                 # Cached API responses
│  └─ runs/                  # Run outputs (artifacts per date/symbol)
├─ docs/                     # Documentation & diagrams
│  └─ architecture.md
├─ notebooks/                  # Jupyter notebooks (experiments & deliverables)
│  ├─ 00_dev_scratch.ipynb     # Sandbox for quick tests and prototyping
│  ├─ 01_data_ingestion.ipynb  # Exploration and validation of data sources (APIs, cleaning, caching)
│  ├─ 02_analysis_sentiment.ipynb # NLP experiments (sentiment, entity extraction, summaries)
│  ├─ 03_system_orchestration.ipynb # Workflow orchestration tests (routing, evaluator–optimizer loop)
│  └─ final_project.ipynb      # Clean, consolidated notebook for submission (exported to PDF/HTML)
├─ src/                      # Source code (modular project implementation)
│  ├─ analysis/              # Analyst agent functions (NLP, sentiment, signals)
│  │  ├─ text.py             # Text cleaning, tokenization, entity extraction <Ali>
│  │  ├─ signals.py          # Generate trading signals from data <Victor>
│  ├─ config/                # Centralized configuration & settings
│  │  ├─ settings.py         # TTLs, paths, feature flags <Ali>
│  ├─ data_io/               # Data ingestion, cleaning, caching (APIs, schemas)
│  │  ├─ prices.py           # Fetch stock prices & volumes via yfinance <Victor>
│  │  ├─ news.py             # Ingest and clean financial news feeds <Victor>
│  │  ├─ cache.py            # Local caching logic with TTL (rate-limit safe) <Victor>
│  ├─ system/                # Orchestration & routing (system engineer tasks)
│  │  ├─ router.py           # Route tasks to specialized agents <Ali>
│  │  ├─ orchestrator.py     # Master orchestrator (run_agent) <Ali>
│  │  ├─ memory.py           # Agent memory across runs (JSON/SQLite) <Ali>
|  └─ agents.py              # All agents are here <Sunitha>
├─ ui/                       # User interface components
│  └─ gradio_app.py          # Gradio library for visualization <Ali>
├─ .env.example              # Example API keys & secrets (do not commit real ones)
├─ requirements.txt          # Python dependencies for the project
├─ README.md                 # Project overview and usage guide
└─ .gitignore                # Ignore cache, envs, temp files, big datasets