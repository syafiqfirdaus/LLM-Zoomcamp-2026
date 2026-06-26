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

### Notes

- The solution uses a local `.env` file only for development and does not track secrets in Git.
- The repo has been cleaned to remove any accidental `.env` secrets from history.
