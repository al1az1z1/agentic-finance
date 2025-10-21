#  Agentic Finance
**AAI-520 Final Project — Multi-Agent Financial Analysis System using Agentic AI**

---

## 📘 Overview
**Agentic Finance** is a collaborative capstone project developed for the University of San Diego’s *AAI-520: Natural Language Processing and Generative AI* course.  
The system demonstrates how **Agentic AI** principles—autonomous planning, routing, evaluation, and iteration—can be applied to **investment research**.

Unlike static scripts, Agentic Finance coordinates multiple **specialized LLM agents** that analyze earnings reports, financial news, market indicators, and risk metrics end-to-end.  
The architecture mirrors the multi-agent stacks used in professional financial research environments.

---

## 🧩 Project Directory Layout

```
agentic-finance/
├─ data/                      # Local data storage
│  ├─ cache/                  # Cached API responses (JSON files with TTL)
│  └─ runs/                   # Run artifacts and output logs per ticker/date
│
├─ notebooks/                 # Jupyter notebooks (experiments & deliverables)
│  └─ AAI-520_Final_Team_project_Group_5.ipynb  # Main consolidated notebook (exported to PDF/HTML)
│
├─ src/                       # Core application source code
│  ├─ analysis/               # Analytical and text-processing helpers
│  │  ├─ features.py          # # Basic Feature Engineering — SMA and RSI Computation
│  │  ├─ text.py              # # News basic Preprocessing Prepare news for the pipeline, Make LLM I/O stable
│  │
│  ├─ config/                 # Centralized configuration & environment settings
│  │  ├─ settings.py          # Paths, API TTLs, feature flags, model defaults  
│  │
│  ├─ data_io/                # Data ingestion & transformation from APIs
│  │  ├─ cache.py             # Manages JSON-based caching with time-to-live (TTL) control to avoid redundant API calls.
│  │  ├─ earnings.py          # Fetches quarterly EPS data from Yahoo Finance using earnings_dates, normalizes columns (Estimate, Reported, Surprise%), and caches results per symbol
│  │  ├─ indicators.py        # Retrieves technical indicators (RSI, SMA, MACD) for price trend analysis.
│  │  ├─ news.py              # Collects and cleans financial news articles or headlines, providing sentiment-ready text 
│  │  ├─ risk.py              # Computes portfolio and stock-level risk metrics (Sharpe ratio, volatility, drawdown, beta, VaR).
│  │
│  ├─ system/                 # Orchestration & routing logic
│  │  ├─ memory.py            # Append agent memories across runs (JSONL) 
│  │  ├─ orchestrator.py      # Master orchestrator coordinating agents  
│  │  ├─ router.py            # Task routing to specialized agents 
│  │
│  └─ agents.py               # Definitions of all LLM agents  <Sunitha>
│
├─ ui/                        # Front-end interface
│  └─ gradio_app.py           # Interactive Gradio dashboard for running analyses  
│
├─ .env.example               # Example API key structure (never commit real keys)
├─ requirements.txt           # Python dependencies
├─ README.md                  # Project overview and setup guide
└─ .gitignore                 # Ignore caches, environments, large datasets
```

---

## System Architecture
**Agentic Workflow Patterns Implemented:**
1. **Prompt-Chaining** – Ingest News → Preprocess → Classify → Extract → Summarize  
2. **Routing** – Direct content to specialized agents (News, Earnings, Risk, Technical).  
3. **Evaluator–Optimizer Loop** – Critique and re-synthesize analysis based on feedback.

**Agents Implemented:**
| Agent | Role |
|-------|------|
|  **NewsAnalysisAgent** | Performs financial news sentiment and impact analysis. |
|  **EarningsAnalysisAgent** | Evaluates company fundamentals via EPS trends and surprise ratios. |
|  **MarketSignalsAgent** | Interprets technical indicators and price momentum. |
|  **RiskAssessmentAgent** | Quantifies volatility, Sharpe ratio, VaR, and portfolio risk. |
|  **SynthesisAgent** | Combines all agent outputs into a single investment recommendation. |
|  **CritiqueAgent** | Reviews synthesis quality, flags biases, and suggests refinements. |

---

##  Technology Stack
- **Python 3.10+**
- **OpenAI GPT-4o / GPT-4o-mini** (LLM backbone)
- **Pandas**, **NumPy**, **yfinance**, **Alpha Vantage**, **FRED API**
- **Gradio** for user interface
- **dotenv** for environment management
- **GitHub** for collaboration and version control

---

##  Setup & Usage

1. **Clone Repository**
   ```bash
   git clone https://github.com/al1az1z1/agentic-finance.git
   cd agentic-finance
   ```

2. **Create Environment & Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   Copy `.env.example` → `.env` and add your API keys:
   ```
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
   ALPHAVANTAGE_API_KEY=xxxxxxxxxxxx
   ```

4. **Run the Notebook**
   Open `notebooks/AAI-520_Final_Team_project_Group_5.ipynb`  
   or launch the Gradio UI:
   ```bash
   python ui/gradio_app.py
   ```

---

## Contributors

| Name | Role | LinkedIn |
|------|------|-----------|
| **Sunitha Kosireddy** | Agent Developer & NLP Core | [linkedin.com/in/sunitha-k-0bb53693](https://www.linkedin.com/in/sunitha-k-0bb53693/) |
| **Victor Salcedo** | Data Engineer & API Integration | [linkedin.com/in/victorjsalcedo](https://www.linkedin.com/in/victorjsalcedo) |
| **Ali Azizi** | System Engineer & UI/Integration Lead | [linkedin.com/in/al1az1z1](https://www.linkedin.com/in/al1az1z1/) |

---

## Acknowledgment
This project was developed as part of the **MS in Applied Artificial Intelligence** program at the **University of San Diego**.  
Special thanks to our instructor for guidance

---

## License
This project is for academic and educational use only.  
© 2025 — Team Agentic Finance, University of San Diego.

---

##  Additional Resources
-  [Final Notebook (HTML)](https://github.com/al1az1z1/agentic-finance/blob/main/notebooks/AAI-520_Final_Team_project_Group_5.html)  
