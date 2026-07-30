# 📈 QuantPulse AI: Quantitative Finance RAG & Agent Assistant

> **LLM Zoomcamp Project Attempt 1 Submission**  
> *Target Domain: Quantitative Finance, Derivatives Pricing, Algorithmic Trading, and Financial Risk Management*

QuantPulse AI is an end-to-end LLM application combining Retrieval-Augmented Generation (RAG) and interactive quantitative finance tools. Built specifically for beginner to intermediate quants, financial analysts, and algorithmic trading enthusiasts, QuantPulse AI indexes a rich corpus of quantitative finance documents and provides clear, math-accurate answers (with LaTeX formulas), interactive Black-Scholes valuation, live market stock data, and comprehensive real-time telemetry.

---

## App Screenshots

### Quant Assistant

![QuantPulse AI Quant Assistant](./quant-assistant.png)

### Black-Scholes Calculator

![QuantPulse AI Black-Scholes Calculator](./black-scholes-calculator.png)

### Live Market Ticker Viewer

![QuantPulse AI Live Market Ticker Viewer](./live-market-ticker-viewer.png)

### Monitoring Dashboard

![QuantPulse AI Monitoring Dashboard](./monitoring-dashboard.png)

### Full Main Dashboard

![QuantPulse AI Main Dashboard](./quantpulse-main-dashboard.png)

---

## 🎯 Problem Statement

Quantitative finance spans complex mathematical concepts (e.g., Black-Scholes PDE, stochastic calculus, GARCH, Delta hedging, Value at Risk). Novice traders and finance students often struggle to translate raw financial formulas into practical trading strategies or evaluate risk metrics accurately.

**QuantPulse AI solves this problem by:**
1. Providing a search-augmented Q&A assistant trained on quantitative finance literature.
2. Formulating response explanations with LaTeX math notation and Python code implementations.
3. Supplying an interactive Black-Scholes valuation calculator and live ticker quotes.
4. Tracking search quality, latency, and user feedback through a telemetry dashboard.

---

## 📁 Repository Structure

```
Project Attempt 1/quant_pulse_ai/
├── app.py                      # Main Streamlit Web Application & UI
├── requirements.txt            # Python Dependencies
├── Dockerfile                  # Container build file
├── docker-compose.yml          # Container orchestration (App + Elasticsearch)
├── README.md                   # Project documentation & evaluation rubric
├── data/
│   ├── quant_finance_kb.json   # Knowledge Base dataset (15 key quant documents)
│   ├── ground_truth.json       # Ground Truth Q&A dataset (30 benchmark items)
│   ├── retrieval_eval_results.json # Retrieval evaluation metrics
│   └── llm_eval_results.json   # LLM answer evaluation scores
└── src/
    ├── ingest.py               # Automated ingestion pipeline script
    ├── search.py               # BM25, Dense Vector, Hybrid RRF, Query Rewriter, Re-ranker
    ├── evaluate_retrieval.py   # Hit Rate @ k & MRR evaluation benchmark
    ├── evaluate_llm.py         # LLM prompt evaluation & LLM-as-a-Judge script
    └── db.py                   # SQLite query logger & user feedback database
```

---

## 🛠️ Technology Stack

- **LLM**: OpenAI `gpt-4o-mini` / `gpt-4o` (or Gemini 1.5)
- **Knowledge Base**: JSON vector-ready corpus + `minsearch` BM25 + `sentence-transformers/all-MiniLM-L6-v2`
- **Retrieval Methods Evaluated**: MinSearch (BM25), Dense Vector Search, Hybrid (RRF), Query Rewriting, and Re-ranking
- **Interface**: Streamlit with interactive Plotly charts and LaTeX math rendering
- **Monitoring & Feedback**: SQLite database + Logfire telemetry + Streamlit dashboard with 5+ charts
- **Containerization**: Docker & Docker Compose

---

## 📊 Evaluation & Benchmark Results

### 1. Retrieval Evaluation (Evaluated on `ground_truth.json` with 30 queries)

| Search Approach | Hit Rate @ 1 | Hit Rate @ 3 | Hit Rate @ 5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **MinSearch (BM25)** | 0.8333 | 0.9333 | 0.9667 | 0.8806 |
| **Vector Search (MiniLM)** | 0.8667 | 0.9667 | 1.0000 | 0.9167 |
| **Hybrid Search (RRF)** | **0.9000** | **1.0000** | **1.0000** | **0.9444** |
| **Hybrid + Query Rewriting** | 0.8667 | 0.9667 | 1.0000 | 0.9194 |
| **Hybrid + Re-ranking** | **0.9333** | **1.0000** | **1.0000** | **0.9611** |

*Conclusion: **Hybrid Search with Re-ranking** achieved the top performance with an MRR of **0.9611** and Hit Rate @ 3 of **100%**.*

### 2. LLM Output Evaluation (LLM-as-a-Judge, Scale 1-5)

| Prompt Strategy | Avg Score (/ 5.0) | Key Observation |
| :--- | :---: | :--- |
| **Simple Prompt (Baseline)** | 4.10 | Answers were correct but concise, lacking mathematical detail. |
| **Quant Expert System Prompt** | **4.90** | Provided LaTeX formulas, parameter definitions, and practical risk insights. |

---

## 🚀 How to Run the Project

### Option A: Running locally with `.venv`

1. Open your terminal in the workspace root.
2. Run the automated ingestion pipeline:
   ```bash
   .\.venv\Scripts\python.exe "Project Attempt 1/quant_pulse_ai/src/ingest.py"
   ```
3. Run the retrieval evaluation benchmark:
   ```bash
   .\.venv\Scripts\python.exe "Project Attempt 1/quant_pulse_ai/src/evaluate_retrieval.py"
   ```
4. Launch the Streamlit application:
   ```bash
   .\.venv\Scripts\python.exe -m streamlit run "Project Attempt 1/quant_pulse_ai/app.py"
   ```
5. Open your browser at `http://localhost:8501`.

### Option B: Running with Docker Compose

1. Build and start the containerized stack:
   ```bash
   docker-compose up --build
   ```
2. Access the application at `http://localhost:8501`.

---

## ✅ Course Peer Review Rubric Checklist

- [x] **Problem Description**: Detailed quantitative finance problem statement and practical application scope.
- [x] **Retrieval Flow**: Knowledge Base + BM25 + Vector Search + LLM.
- [x] **Retrieval Evaluation**: Evaluated 5 retrieval approaches (MinSearch, Vector, Hybrid RRF, Rewriting, Re-ranking) reporting Hit Rate & MRR.
- [x] **LLM Evaluation**: Evaluated baseline prompt vs Expert System prompt with LLM-as-a-Judge scoring.
- [x] **Interface**: Full interactive Streamlit Web UI with LaTeX formulas, option pricing calculator, live ticker viewer, and source expansion.
- [x] **Ingestion Pipeline**: Automated ingestion script (`ingest.py`) parsing JSON documents and building vector indices.
- [x] **Monitoring**: User feedback collection (Thumbs Up/Down) persisted in SQLite database + Analytics dashboard with 5+ charts.
- [x] **Containerization**: `Dockerfile` and `docker-compose.yml` provided for full stack execution.
- [x] **Reproducibility**: Clear instructions, specified dependencies (`requirements.txt`), and automated dataset.
- [x] **Best Practices**:
  - [x] Hybrid search (BM25 + Dense Vector RRF)
  - [x] Document re-ranking
  - [x] User query rewriting
