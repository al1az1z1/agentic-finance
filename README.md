# agentic-finance
AAI-520 Final Project: Multi-Agent Financial Analysis System using Agentic AI

## Project Directory Layout

agentic-finance/
├─ notebooks/                  # Jupyter notebooks (experiments & deliverables)
│  ├─ 00_dev_scratch.ipynb     # Sandbox for quick tests and prototyping
│  ├─ 01_data_ingestion.ipynb  # Exploration and validation of data sources (APIs, cleaning, caching)
│  ├─ 02_analysis_sentiment.ipynb # NLP experiments (sentiment, entity extraction, summaries)
│  ├─ 03_system_orchestration.ipynb # Workflow orchestration tests (routing, evaluator–optimizer loop)
│  └─ final_project.ipynb      # Clean, consolidated notebook for submission (exported to PDF/HTML)
├─ src/                      # Source code (modular project implementation)
│  ├─ data_io/               # Data ingestion, cleaning, caching (APIs, schemas)
│  │  ├─ prices.py           # Fetch stock prices & volumes via yfinance
│  │  ├─ news.py             # Ingest and clean financial news feeds
│  │  ├─ macro.py            # Macroeconomic data ingestion (FRED)
│  │  ├─ filings.py          # Company filings pipeline (SEC EDGAR)
│  │  ├─ cache.py            # Local caching logic with TTL (rate-limit safe)
│  │  └─ errors.py           # Custom error codes (NoDataError, RateLimitError)
│  ├─ analysis/              # Analyst agent functions (NLP, sentiment, signals)
│  │  ├─ text.py             # Text cleaning, tokenization, entity extraction
│  │  ├─ sentiment.py        # Sentiment classification functions
│  │  ├─ extract.py          # Extract events, risks, guidance from text
│  │  ├─ summarize.py        # Summarize financial impacts
│  │  ├─ signals.py          # Generate trading signals from data
│  │  └─ api.py              # Unified interface: analyze_symbol, compare_symbols
│  ├─ system/                # Orchestration & routing (system engineer tasks)
│  │  ├─ router.py           # Route tasks to specialized agents
│  │  ├─ orchestrator.py     # Master orchestrator (run_agent)
│  │  ├─ memory.py           # Agent memory across runs (JSON/SQLite)
│  │  └─ reporting.py        # Save outputs: briefs, evidence, metrics
│  ├─ vecstore/              # Vector database adapters for embeddings
│  │  ├─ base.py             # Base interface for vector DB operations
│  │  └─ chroma.py           # Example vector DB implementation (Chroma/FAISS)
│  ├─ config/                # Centralized configuration & settings
│  │  ├─ settings.py         # TTLs, paths, feature flags
│  │  └─ __init__.py         # Package marker
│  └─ contracts.md           # Documentation of input/output schemas & APIs
├─ ui/                       # User interface components
│  └─ app.py                 # Streamlit dashboard for visualization
├─ data/                     # Local data storage
│  ├─ cache/                 # Cached API responses
│  └─ runs/                  # Run outputs (artifacts per date/symbol)
├─ tests/                    # Optional unit tests (schema & functionality checks)
│  ├─ test_data_io.py
│  ├─ test_analysis.py
│  └─ test_system.py
├─ docs/                     # Documentation & diagrams
│  └─ architecture.md
├─ .env.example              # Example API keys & secrets (do not commit real ones)
├─ requirements.txt          # Python dependencies for the project
├─ README.md                 # Project overview and usage guide
└─ .gitignore                # Ignore cache, envs, temp files, big datasets