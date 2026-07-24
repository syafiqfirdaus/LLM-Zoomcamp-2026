# Homework 5 Answers

### Q1. First trace
**Question:** How many spans does the trace produce?
**Answer:** 3

### Q2. Capturing metrics as span attributes
**Question:** How many input tokens do we see?
**Answer:** 7000

### Q3. Span timing
**Question:** For a typical query, roughly how long does the LLM call take?
**Answer:** Over 2000ms

### Q4. Saving traces to SQLite
**Question:** Which span names appear in the `spans` table?
**Answer:** `rag`, `search`, and `llm`

### Q5. Querying trace data
**Question:** Which span type takes the most total time (excluding rag)?
**Answer:** `llm`

### Q6. Token stability across runs
**Question:** How much do the input tokens vary across these 4 runs?
**Answer:** They're identical
