# Data Ingestion Module

This branch contains the **Data Ingestion** component of our Multi-Agent Financial Analysis System.  
It prepares **clean, ready-to-use Hugging Face datasets** from financial APIs so that downstream agents can perform research, analysis, and reasoning.  

---

## Files
- `DataIngestion.ipynb`  
  Jupyter Notebook with runnable examples and sample outputs (for testing and grading).  
- `data_ingestion.py`  
  Pure Python module with the reusable ingestion logic (`AlphaConnector`, `DataIngestionManager`, `to_hf`).  
  Agents should import this file when they need live financial or news data.  

---

## Features
- **Yahoo Finance**  
  - Daily prices (open, high, low, close, volume)  
- **Alpha Vantage** (requires API key)  
  - News sentiment (`NEWS_SENTIMENT`)  
  - Technical Indicators (e.g., SMA, RSI; MACD requires premium)  

All outputs are converted into **Hugging Face Datasets** for easy integration with agents and ML pipelines.

---

## Setup

1. **Install dependencies** (in Colab or local environment):
   ```bash
   pip install yfinance datasets requests
   
2. **Set up Alpha Vantage API key** (once per runtime):
   ```
   import os
   os.environ["ALPHAVANTAGE_API_KEY"] = "YOUR_KEY_HERE"

   # If your code expects this name instead, use:
   # os.environ["ALPHA_VANTAGE_KEY"] = "YOUR_KEY_HERE"

3. **Import and initialize the Data Manager**:
  ```
   from data_ingestion import DataIngestionManager

   mgr = DataIngestionManager()  # picks up the key from os.environ


