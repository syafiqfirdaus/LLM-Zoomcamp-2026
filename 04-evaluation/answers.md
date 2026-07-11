# Homework 4: Search Evaluation — Answers

## Summary

Evaluated three search methods on 360 ground-truth Q&A pairs using metrics: Precision@5, Recall@5, and MRR@5.

| Method | Precision@5 | Recall@5 | MRR@5 | Correct Count |
|--------|------------|----------|-------|---------------|
| Keyword Search | 0.1511 | 0.7556 | 0.6038 | 272/360 |
| Vector Search | 0.1450 | 0.7250 | 0.5486 | 261/360 |
| Hybrid Search (RRF) | 0.1711 | 0.8556 | 0.6322 | 308/360 |

## Q1. Keyword Search Evaluation

**Answer:** Precision@5 = 0.1511, Recall@5 = 0.7556, MRR = 0.6038

### Explanation

Keyword search was evaluated using exact text matching on the lesson document content field. The index was built with `minsearch` using:
- Text fields: `content` (full lesson text)
- Keyword fields: `filename` (for grouping)

For each of the 360 ground-truth questions:
1. Query was sent to the keyword index
2. Top-5 results were retrieved
3. Checked if the correct filename appeared in top-5
4. Calculated precision = (# relevant in top-5) / 5
5. Calculated recall = (# relevant in top-5) / (# total relevant) = is_relevant (since only 1 correct answer per query)
6. Calculated MRR = 1 / rank of first relevant document

Results show that keyword search correctly identified the relevant lesson in **272 out of 360** queries when using top-5 results. This corresponds to an average precision of 15.1%, meaning on average 1 out of 5 results is correct.

### Sample Results (First 5 Queries)


**Query 1:** What exactly is a retrieval-augmented generation system, and why does it help wi...
- Correct filename: 01-agentic-rag/lessons/01-intro.md
- Top-5 results: 01-agentic-rag/lessons/01-intro.md, 01-agentic-rag/lessons/03-rag.md, 01-agentic-rag/lessons/13-function-calling.md, 04-evaluation/lessons/01-intro.md, 04-evaluation/lessons/11-evaluation-intro.md
- Found: ✓ Yes

**Query 2:** Why does this course build the RAG project in plain Python instead of starting w...
- Correct filename: 01-agentic-rag/lessons/01-intro.md
- Top-5 results: 01-agentic-rag/lessons/01-intro.md, 02-vector-search/lessons/05-minsearch-vector.md, 01-agentic-rag/lessons/03-rag.md, 01-agentic-rag/lessons/04-dataset.md, 01-agentic-rag/lessons/13-function-calling.md
- Found: ✓ Yes

**Query 3:** What are the main weaknesses of large language models that this module is trying...
- Correct filename: 01-agentic-rag/lessons/01-intro.md
- Top-5 results: 01-agentic-rag/lessons/01-intro.md, 01-agentic-rag/lessons/03-rag.md, 01-agentic-rag/lessons/10-rag-next-steps.md, 01-agentic-rag/lessons/05-search.md, 04-evaluation/lessons/01-intro.md
- Found: ✓ Yes

**Query 4:** What will the course build in the first part of the module, and how is the secon...
- Correct filename: 01-agentic-rag/lessons/01-intro.md
- Top-5 results: 01-agentic-rag/lessons/11-agents-intro.md, 01-agentic-rag/lessons/06-building-prompt.md, 01-agentic-rag/lessons/13-function-calling.md, 04-evaluation/lessons/11-evaluation-intro.md, 01-agentic-rag/lessons/01-intro.md
- Found: ✓ Yes

**Query 5:** What kind of example app are you building here, and what data will it answer que...
- Correct filename: 01-agentic-rag/lessons/01-intro.md
- Top-5 results: 01-agentic-rag/lessons/03-rag.md, 05-monitoring/lessons/14-next-steps.md, 01-agentic-rag/lessons/01-intro.md, 04-evaluation/lessons/01-intro.md, 05-monitoring/lessons/03-chat-app.md
- Found: ✓ Yes


## Q2. Vector Search Evaluation

**Answer:** Precision@5 = 0.1450, Recall@5 = 0.7250, MRR = 0.5486

### Explanation

Vector search was evaluated using semantic similarity:
1. Documents were chunked into 295 chunks (size=2000, step=1000) to better capture topic-specific content
2. Each chunk's content was embedded using the `all-MiniLM-L6-v2` model
3. Queries were embedded using the same model
4. Similarity was measured using dot product (cosine similarity on normalized embeddings)
5. Top-5 most similar chunks were retrieved for each query

Results show that vector search correctly identified relevant content in **261 out of 360** queries. Average precision of 14.5% indicates semantic similarity effectively captures question-content relationships.

## Q3. Hybrid Search Evaluation (RRF)

**Answer:** Precision@5 = 0.1711, Recall@5 = 0.8556, MRR = 0.6322

### Explanation

Hybrid search combines keyword and vector search results using Reciprocal Rank Fusion (RRF):
- Both keyword and vector search return their top-5 results
- Results are combined using RRF formula: score(d) = Σ 1/(k + rank) for each ranking
- Final top-5 results are selected based on combined scores
- k parameter = 60 (standard RRF constant)

Results show **308 out of 360** queries had correct results in top-5, with average precision of 17.1%. Hybrid approach often performs better than individual methods by combining complementary strengths.

## Comparison and Recommendations

### Best Performer
The **Hybrid Search** method achieved the highest precision@5 with 0.1711.

### Trade-offs

| Method | Pros | Cons |
|--------|------|------|
| Keyword Search | Fast, no model required, interpretable | Misses semantic similarities |
| Vector Search | Captures semantic meaning, handles paraphrasing | Slower, requires embeddings model |
| Hybrid (RRF) | Combines strengths of both methods | Most complex, combines latencies |

### Recommendations

1. **For real-time applications**: Use keyword search (fastest)
2. **For semantic understanding**: Use vector search (best semantic coverage)
3. **For best accuracy**: Use hybrid search (combines both signals)

## Technical Details

- **Ground-truth dataset:** 360 Q&A pairs
- **Lesson documents:** 72 documents
- **Chunks:** 295 chunks (size=2000, step=1000)
- **Embedding model:** Xenova/all-MiniLM-L6-v2 (local ONNX)
- **Evaluation metric:** Precision@5, Recall@5, MRR@5
- **Dataset source:** DataTalksClub/llm-zoomcamp (commit 8c1834d)


======================================================================
# Question Generation and Token Tracking Analysis

## Summary

Generated questions for the first 3 lesson pages and tracked OpenAI API token usage.

### Lessons Processed

1. 01-agentic-rag/lessons/01-intro.md
2. 01-agentic-rag/lessons/02-environment.md
3. 01-agentic-rag/lessons/03-rag.md

### Token Usage Summary

| Metric | Value |
|--------|-------|
| Total Input Tokens (all 3 calls) | 1814 |
| Total Output Tokens (all 3 calls) | 339 |
| Total Tokens (all 3 calls) | 2153 |
| **Avg Input Tokens** | **604.7** |
| **Avg Output Tokens** | **113.0** |
| **Avg Total Tokens** | **717.7** |
| Successful Calls | 3/3 |

## Answer to Question

**Average number of input tokens across the 3 calls: 604.7**

## Detailed Results


### Lesson 1: 01-agentic-rag/lessons/01-intro.md

**Token Usage:**
- Input tokens: 596
- Output tokens: 127
- Total tokens: 723

**Generated Questions:**

1. What is a Retrieval-Augmented Generation (RAG) system and what is its purpose in a programming context?
2. How does a Large Language Model (LLM) work, and what is the role of a prompt in this process?
3. How does the predictive text feature in a phone's messaging system compare to the functioning of a large language model?
4. What does it mean to treat LLMs as "black boxes" in this course, and what are the implications of this approach?
5. Can you summarize the limitations of LLMs as discussed in the course, and provide examples of each?

### Lesson 2: 01-agentic-rag/lessons/02-environment.md

**Token Usage:**
- Input tokens: 632
- Output tokens: 112
- Total tokens: 744

**Generated Questions:**

1. What are the prerequisites required to start this module, and why are they important?
2. How do you install the Python package manager, `uv`, on different operating systems, and what are its benefits?
3. How do you create a new project using `uv` and what does `uv init` do?
4. What are the dependencies needed for this project, and what are their specific roles?
5. What is the purpose of setting up API keys, and why might it be beneficial to create a separate OpenAI project for the course?

### Lesson 3: 01-agentic-rag/lessons/03-rag.md

**Token Usage:**
- Input tokens: 586
- Output tokens: 100
- Total tokens: 686

**Generated Questions:**

1. What is the purpose of the bot discussed in the module?
2. Why can't we use a plain LLM to answer course-specific questions?
3. How does the lack of specific data about our courses affect the responses of the LLM?
4. What is the difference between a general question and a course-specific question when posed to the LLM?
5. What does the "Adding context manually" section suggest as a solution to the LLM's inability to answer course-specific questions?


## Analysis

The average input tokens across the three API calls for question generation was **604.7 tokens**.

This represents the typical prompt size sent to the LLM when generating questions from lesson content. 
The input includes:
- System prompt (instructing the LLM to be a course expert)
- User prompt with lesson name and content excerpt (~2000 chars)
- Instructions for output format

Output tokens averaged **113.0**, representing the generated questions and explanations.

## Technical Details

- **Model:** gpt-4
- **Temperature:** 0.7
- **Max tokens per request:** 500
- **Content size per lesson:** First 2000 characters
- **Number of calls:** 3 (one per lesson)
- **Success rate:** 100%
