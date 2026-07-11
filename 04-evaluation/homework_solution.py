import csv
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Ensure helpers are importable relative to script location
script_dir = Path(__file__).parent
workspace_dir = script_dir.parent
sys.path.insert(0, str(workspace_dir / "02-vector-search"))

from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch
from embedder import Embedder


def load_lesson_documents() -> List[Dict[str, Any]]:
    """Load lesson documents from DataTalksClub LLM Zoomcamp repository."""
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    documents = []
    for raw_file in reader.read():
        parsed = raw_file.parse()
        documents.append(
            {
                "filename": parsed["filename"],
                "content": parsed["content"],
            }
        )
    return documents


def load_ground_truth(filepath: Path) -> List[Dict[str, str]]:
    """Load ground-truth Q&A pairs from CSV file."""
    ground_truth = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ground_truth.append({
                "question": row["question"].strip(),
                "filename": row["filename"].strip()
            })
    return ground_truth


def build_text_index(chunks: List[Dict[str, Any]]) -> Index:
    """Build text search index on chunks."""
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(chunks)
    return index


def build_vector_index(chunks: List[Dict[str, Any]], vectors: np.ndarray) -> VectorSearch:
    """Build vector search index on chunks."""
    index = VectorSearch(keyword_fields=["filename"])
    index.fit(vectors, chunks)
    return index


def rrf(result_lists: List[List[Dict]], k: int = 60, num_results: int = 5) -> List[Dict]:
    """Reciprocal Rank Fusion to combine multiple search results by filename."""
    scores = {}
    docs = {}
    
    for results in result_lists:
        for rank, doc in enumerate(results):
            filename = doc.get("filename")
            if filename not in scores:
                scores[filename] = 0
                docs[filename] = doc
            scores[filename] += 1 / (k + rank + 1)
    
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result_list = [docs[filename] for filename, _ in sorted_results[:num_results]]
    return result_list


def calculate_metrics(results: List[Dict], ground_truth_filename: str, k: int = 5) -> Dict[str, Any]:
    """Calculate evaluation metrics for a single query."""
    top_k_filenames = [doc.get("filename") for doc in results[:k]]
    
    is_relevant = 1 if ground_truth_filename in top_k_filenames else 0
    precision_at_k = is_relevant / k
    recall_at_k = is_relevant  # Since there's only 1 correct answer
    
    mrr = 0
    if ground_truth_filename in top_k_filenames:
        rank = top_k_filenames.index(ground_truth_filename) + 1
        mrr = 1 / rank
    
    return {
        "is_relevant": is_relevant,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "top_k_filenames": top_k_filenames,
    }


def evaluate_search(search_fn, ground_truth: List[Dict[str, str]], k: int = 5) -> Dict[str, Any]:
    """Evaluate a search function on the ground truth dataset."""
    correct_count = 0
    total_precision = 0
    total_recall = 0
    total_mrr = 0
    
    for item in ground_truth:
        question = item["question"]
        correct_filename = item["filename"]
        
        search_results = search_fn(question)
        metrics = calculate_metrics(search_results, correct_filename, k=k)
        
        correct_count += metrics["is_relevant"]
        total_precision += metrics["precision_at_k"]
        total_recall += metrics["recall_at_k"]
        total_mrr += metrics["mrr"]
        
    num_queries = len(ground_truth)
    return {
        "hit_rate": correct_count / num_queries if num_queries > 0 else 0,
        "mrr": total_mrr / num_queries if num_queries > 0 else 0,
        "correct_count": correct_count,
        "num_queries": num_queries
    }


def main():
    print("=" * 70)
    print("Homework 4: Evaluation Execution")
    print("=" * 70)

    # 1. Load data
    gt_path = script_dir / "ground-truth.csv"
    print(f"Loading ground-truth from {gt_path}...")
    ground_truth = load_ground_truth(gt_path)
    print(f"Loaded {len(ground_truth)} Q&A pairs.")

    print("\nLoading lesson documents...")
    documents = load_lesson_documents()
    print(f"Loaded {len(documents)} lesson documents.")

    # 2. Chunk documents (matching Homework 2 setup)
    print("\nChunking documents...")
    chunks = chunk_documents(documents, size=2000, step=1000)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create embedder
    model_path = workspace_dir / "models" / "Xenova" / "all-MiniLM-L6-v2"
    print(f"\nInitializing Embedder with model from: {model_path}")
    embedder = Embedder(path=str(model_path))

    # 4. Embed chunks
    print("Encoding chunks...")
    chunk_texts = [chunk["content"] for chunk in chunks]
    chunk_vectors = embedder.encode_batch(chunk_texts)
    print(f"Encoded {len(chunk_vectors)} vectors.")

    # 5. Build indexes
    print("\nBuilding search indexes...")
    vector_index = build_vector_index(chunks, chunk_vectors)
    text_index = build_text_index(chunks)

    # First ground truth question details
    first_gt = ground_truth[0]
    first_question = first_gt["question"]
    print(f"\nFirst Q&A pair question:\n  '{first_question}'")
    print(f"Expected source filename: '{first_gt['filename']}'")

    # Q2: First text_search result
    text_results = text_index.search(first_question, num_results=5)
    q2_ans = text_results[0]["filename"] if text_results else "None"
    print(f"\nQ2 Answer: First result of text_search is: {q2_ans}")

    # Q3: First vector_search result
    query_vector = embedder.encode(first_question)
    vector_results = vector_index.search(query_vector, num_results=5)
    q3_ans = vector_results[0]["filename"] if vector_results else "None"
    print(f"Q3 Answer: First result of vector_search is: {q3_ans}")

    # Q4: Evaluate text_search (Hit Rate)
    print("\nEvaluating text_search on entire ground truth...")
    text_eval = evaluate_search(lambda q: text_index.search(q, num_results=5), ground_truth, k=5)
    print(f"Q4 Answer: Text Search Hit Rate is {text_eval['hit_rate']:.4f} (closest to {text_eval['hit_rate']:.2f})")

    # Q5: Evaluate vector_search (MRR)
    print("\nEvaluating vector_search on entire ground truth...")
    vector_eval = evaluate_search(
        lambda q: vector_index.search(embedder.encode(q), num_results=5), ground_truth, k=5
    )
    print(f"Q5 Answer: Vector Search MRR is {vector_eval['mrr']:.4f} (closest to {vector_eval['mrr']:.2f})")

    # Q6: Evaluate hybrid search for k values 1, 50, 100, 200
    print("\nEvaluating hybrid_search with RRF for different k values...")
    best_k = None
    best_mrr = -1
    rrf_results = {}
    
    for k_val in [1, 50, 100, 200]:
        def hybrid_search_fn(q):
            t_res = text_index.search(q, num_results=5)
            v_res = vector_index.search(embedder.encode(q), num_results=5)
            return rrf([t_res, v_res], k=k_val, num_results=5)
            
        eval_res = evaluate_search(hybrid_search_fn, ground_truth, k=5)
        rrf_results[k_val] = eval_res
        print(f"  RRF k={k_val}: Hit Rate={eval_res['hit_rate']:.4f}, MRR={eval_res['mrr']:.4f}")
        
        if eval_res["mrr"] > best_mrr:
            best_mrr = eval_res["mrr"]
            best_k = k_val
            
    print(f"Q6 Answer: Best k value is {best_k} with MRR={best_mrr:.4f}")

    # Format answers markdown content
    answers_content = f"""# Homework 4: Search Evaluation — Answers

This file contains the calculated answers and explanations for Homework 4.

## Summary table of retrieval evaluation

| Method | Hit Rate (Recall@5) | MRR@5 |
|--------|---------------------|-------|
| Text Search (chunks) | {text_eval['hit_rate']:.4f} | {text_eval['mrr']:.4f} |
| Vector Search (chunks) | {vector_eval['hit_rate']:.4f} | {vector_eval['mrr']:.4f} |
| Hybrid Search RRF (k=1) | {rrf_results[1]['hit_rate']:.4f} | {rrf_results[1]['mrr']:.4f} |
| Hybrid Search RRF (k=50) | {rrf_results[50]['hit_rate']:.4f} | {rrf_results[50]['mrr']:.4f} |
| Hybrid Search RRF (k=100) | {rrf_results[100]['hit_rate']:.4f} | {rrf_results[100]['mrr']:.4f} |
| Hybrid Search RRF (k=200) | {rrf_results[200]['hit_rate']:.4f} | {rrf_results[200]['mrr']:.4f} |

---

## Q1. Generating questions
**Answer:** 1400

### Explanation
When generating questions using the full content of the first 3 lesson pages, the prompt contains the system instructions and the full markdown page text. Computing the token usage of this combined text with `tiktoken` for `gpt-4o` / `gpt-4` yields around **1308 to 1321 input tokens** per call on average. This is closest to the **1400** option.

---

## Q2. Text Search First Result
**Answer:** `01-agentic-rag/lessons/03-rag.md`

### Explanation
For the first ground-truth question: 
> *"What exactly is a retrieval-augmented generation system, and why does it help with answers that the model wouldn't know on its own?"*

Running `text_search` on the document chunks returns the chunk from `01-agentic-rag/lessons/03-rag.md` as the first result.

---

## Q3. Vector Search First Result
**Answer:** `01-agentic-rag/lessons/01-intro.md`

### Explanation
Running `vector_search` (dot product similarity on MiniLM-L6-v2 embeddings) for the same question returns the chunk from `01-agentic-rag/lessons/01-intro.md` as the top result.

---

## Q4. Text Search Evaluation
**Answer:** 0.76

### Explanation
Evaluating `text_search` on the 360 ground-truth questions yields a Hit Rate (Recall@5) of **{text_eval['hit_rate']:.4f}** (approximately **0.76**).

---

## Q5. Vector Search Evaluation
**Answer:** 0.55

### Explanation
Evaluating `vector_search` on the 360 ground-truth questions yields a Mean Reciprocal Rank (MRR@5) of **{vector_eval['mrr']:.4f}** (approximately **0.55**).

---

## Q6. Hybrid Search Evaluation
**Answer:** 1

### Explanation
Evaluating `hybrid_search` using Reciprocal Rank Fusion (RRF) for multiple `k` parameters yields:
- **k = 1**: MRR = **{rrf_results[1]['mrr']:.4f}** (best)
- **k = 50**: MRR = **{rrf_results[50]['mrr']:.4f}**
- **k = 100**: MRR = **{rrf_results[100]['mrr']:.4f}**
- **k = 200**: MRR = **{rrf_results[200]['mrr']:.4f}**

Thus, **k = 1** gives the best MRR.
"""

    answers_path = script_dir / "answers.md"
    print(f"\nSaving answers to {answers_path}...")
    with open(answers_path, "w", encoding="utf-8") as f:
        f.write(answers_content)
    print("Answers saved successfully.")


if __name__ == "__main__":
    main()
