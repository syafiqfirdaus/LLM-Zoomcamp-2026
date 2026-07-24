# LLM-Zoomcamp-2026

## Homework: 01-agentic-rag

This repository includes the homework solution for the `01-agentic-rag` module.

- `01-agentic-rag/homework_solution.py`: Python implementation for downloading lesson markdown from the Zoomcamp course repo, indexing the documents with `minsearch`, building a RAG flow, and supporting chunking and OpenAI usage tracking.
- `01-agentic-rag/answers.md`: concise answers for homework questions Q1–Q6.
- `01-agentic-rag/requirements.txt`: required Python packages for running the homework solution.

## Homework: 02-vector-search

This repository also includes the homework solution for the `02-vector-search` module.

- `02-vector-search/homework_solution.py`: Python implementation for loading course lesson markdown, chunking content, encoding text with the ONNX `Embedder`, running vector and text search with `minsearch`, and combining results with Reciprocal Rank Fusion.
- `02-vector-search/homework_answers.md`: final computed answers for Homework 2 plus a short explanation of each result.
- `02-vector-search/download.py`: helper script to download the ONNX embedding model.
- `02-vector-search/embedder.py`: ONNX embedder implementation used by the solution.
- `02-vector-search/homework.md`: module homework instructions and answer checklist.

## Homework: 03-orchestration

This repository also includes the completed homework notes for the orchestration module.

- `03-orchestration/homework_answers.md`: concise answers and short explanations for Homework 3, based on the local Kestra runs.
- The homework covers a few small Kestra flows for chat without RAG, chat with RAG, and a simple agent workflow that summarizes text and logs token usage.
- It also includes a short note on how prompt design and summary length affect output size and cost.

## Homework: 04-evaluation

Search evaluation homework comparing text, vector, and hybrid (RRF) search over 360 ground-truth Q&A pairs.

- `04-evaluation/homework_solution.py`: evaluates keyword, vector, and hybrid (RRF) search methods.
- `04-evaluation/answers.md`: evaluation results (Hit Rate, MRR) and homework answers.

| Method | Hit Rate | MRR |
|--------|----------|-----|
| Keyword Search | 0.7583 | 0.5943 |
| Vector Search | 0.7250 | 0.5486 |
| **Hybrid Search (RRF k=1)** | **0.8556** | **0.6486** |

## Homework: 05-monitoring

This repository includes the completed homework for the monitoring module.

- `05-monitoring/hw5.py`: Python script to instrument the RAG pipeline with OpenTelemetry, capture token usage and costs, and export traces to a local SQLite database.
- `05-monitoring/answers.md`: Answers to the homework questions based on the script's execution.

### Notes

- The solution uses a local `.env` file only for development and does not track secrets in Git.
- The repo has been cleaned to remove any accidental `.env` secrets from history.
