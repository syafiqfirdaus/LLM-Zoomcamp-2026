# Homework 4: Search Evaluation

In this homework, we evaluate keyword, vector, and hybrid search over the course lessons. We generate a ground truth dataset and use it to evaluate search methods.

## Setup

This homework continues from homework 2. We reuse the same chunks and the same search functions, so it's easiest to keep working in the same project.

## Q1. Generating questions

When generating questions for the first 3 lesson pages, what is the average number of input tokens across these 3 calls?

- [ ] 140
- [ ] 1400
- [ ] 14000
- [ ] 140000

## Q2. Text Search First Result

After running `text_search` for the first ground truth question, what is the filename of the first result?

- [ ] 01-agentic-rag/lessons/01-intro.md
- [x] 01-agentic-rag/lessons/03-rag.md
- [ ] 01-agentic-rag/lessons/13-function-calling.md
- [ ] 01-agentic-rag/lessons/10-rag-next-steps.md

## Q3. Vector Search First Result

After running `vector_search` for the same question, what is the filename of the first result?

- [x] 01-agentic-rag/lessons/01-intro.md
- [ ] 01-agentic-rag/lessons/03-rag.md
- [ ] 04-evaluation/lessons/11-evaluation-intro.md
- [ ] 04-evaluation/lessons/12-rag-answers.md

## Q4. Text Search Evaluation

After evaluating `text_search` on the ground truth, what is the Hit Rate?

- [ ] 0.55
- [ ] 0.66
- [x] 0.76
- [ ] 0.88

## Q5. Vector Search Evaluation

After evaluating `vector_search` on the ground truth, what is the MRR?

- [ ] 0.35
- [ ] 0.45
- [x] 0.55
- [ ] 0.65

## Q6. Hybrid Search Evaluation

When evaluating `hybrid_search` for k values 1, 50, 100, and 200, which k gives the best MRR?

- [x] 1
- [ ] 50
- [ ] 100
- [ ] 200
