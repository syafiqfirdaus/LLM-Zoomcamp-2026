# Homework 2 Answers

Based on `02-vector-search/homework_solution.py` execution.

## Final Answers

- Q1: -0.02058203437252893
- Q2: 0.36107027225589694
- Q3: `02-vector-search/lessons/07-sqlitesearch-vector.md`
- Q4: `04-evaluation/lessons/05-search-metrics.md`
- Q5: `02-vector-search/lessons/08-pgvector.md`
- Q6: `01-agentic-rag/lessons/13-function-calling.md`

## Explanation

- Q1: The query was embedded using the ONNX `Embedder`, and the first component of the normalized vector was printed.
- Q2: The page content from `02-vector-search/lessons/07-sqlitesearch-vector.md` was embedded and the cosine similarity with the Q1 query vector was computed via dot product.
- Q3: All chunks were embedded, scored against the Q1 query vector, and the highest-scoring chunk's filename was selected.
- Q4: `minsearch.VectorSearch` was used with the query "What metric do we use to evaluate a search engine?" and the top returned filename was reported.
- Q5: The same chunks were indexed with `minsearch.Index` for text search and compared to the vector search top 5 results; the filename present only in the vector results was identified.
- Q6: The query "How do I give the model access to tools?" was searched with both vector and text methods, and results were merged using Reciprocal Rank Fusion (RRF) to determine the top hybrid result.
