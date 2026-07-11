# Homework 4: Search Evaluation — Answers

## Evaluation Summary

Evaluated on 360 ground-truth Q&A pairs generated from course lessons.

| Method | Hit Rate | MRR |
|---|---|---|
| Text Search | 0.7583 | 0.5943 |
| Vector Search | 0.7250 | 0.5486 |
| Hybrid RRF (k=1) | 0.8556 | 0.6486 |
| Hybrid RRF (k=50) | 0.8556 | 0.6322 |
| Hybrid RRF (k=100) | 0.8556 | 0.6322 |
| Hybrid RRF (k=200) | 0.8556 | 0.6322 |

---

## Q1. Generating Questions

**Answer: 1400**

Actual measured average input tokens across 3 calls: **1147.0**  
This is closest to the **1400** option.

| Lesson | Input Tokens |
|---|---|
| 01-intro.md | 861 |
| 02-environment.md | 1080 |
| 03-rag.md | 1500 |
| **Average** | **1147.0** |

---

## Q2. Text Search First Result

**Answer: `01-agentic-rag/lessons/03-rag.md`**

Running `text_search` on the first ground-truth question returns `03-rag.md` as the top result.

---

## Q3. Vector Search First Result

**Answer: `01-agentic-rag/lessons/01-intro.md`**

Running `vector_search` with MiniLM-L6-v2 embeddings on the same question returns `01-intro.md` as the top result.

---

## Q4. Text Search Evaluation

**Answer: 0.76**

Text search Hit Rate (Recall@5) = **0.7583 ≈ 0.76**

---

## Q5. Vector Search Evaluation

**Answer: 0.55**

Vector search MRR@5 = **0.5486 ≈ 0.55**

---

## Q6. Hybrid Search Evaluation

**Answer: k = 1**

| k | MRR |
|---|---|
| 1 | **0.6486** |
| 50 | 0.6322 |
| 100 | 0.6322 |
| 200 | 0.6322 |

**k = 1** gives the best MRR.
