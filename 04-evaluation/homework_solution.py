import csv
import numpy as np
from typing import List, Dict, Any, Set
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch

try:
    from embedder import Embedder
except ImportError:
    try:
        from sys import path as syspath
        syspath.insert(0, '../02-vector-search')
        from embedder import Embedder
    except ImportError as exc:
        raise ImportError(
            "Missing embedder helper. Place `embedder.py` in `02-vector-search/` "
            "or install the course helper script from the upstream repo."
        ) from exc


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


def load_ground_truth(filepath: str = "ground-truth.csv") -> List[Dict[str, str]]:
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


def build_keyword_index(documents: List[Dict[str, Any]]) -> Index:
    """Build keyword search index."""
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(documents)
    return index


def build_vector_index(chunks: List[Dict[str, Any]], vectors: np.ndarray) -> VectorSearch:
    """Build vector search index."""
    index = VectorSearch(keyword_fields=["filename"])
    index.fit(vectors, chunks)
    return index


def build_text_index(chunks: List[Dict[str, Any]]) -> Index:
    """Build text search index for hybrid search."""
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(chunks)
    return index


def rrf(result_lists: List[List[Dict]], k: int = 60, num_results: int = 5) -> List[Dict]:
    """Reciprocal Rank Fusion to combine multiple search results."""
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
    
    # Precision@K: Is the correct document in top-K?
    is_relevant = 1 if ground_truth_filename in top_k_filenames else 0
    precision_at_k = is_relevant / k
    
    # Recall@K: Among top-K results, how many are the correct one
    recall_at_k = is_relevant  # Since there's only 1 correct answer
    
    # MRR (Mean Reciprocal Rank)
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


def evaluate_keyword_search(index: Index, ground_truth: List[Dict[str, str]], k: int = 5) -> Dict[str, Any]:
    """Evaluate keyword search method."""
    results = []
    correct_count = 0
    total_precision = 0
    total_recall = 0
    total_mrr = 0
    
    for idx, item in enumerate(ground_truth):
        question = item["question"]
        correct_filename = item["filename"]
        
        # Search
        search_results = index.search(question, num_results=k)
        
        # Calculate metrics
        metrics = calculate_metrics(search_results, correct_filename, k=k)
        
        results.append({
            "question": question,
            "correct_filename": correct_filename,
            "metrics": metrics,
            "top_k_filenames": metrics["top_k_filenames"]
        })
        
        correct_count += metrics["is_relevant"]
        total_precision += metrics["precision_at_k"]
        total_recall += metrics["recall_at_k"]
        total_mrr += metrics["mrr"]
    
    num_queries = len(ground_truth)
    avg_precision = total_precision / num_queries if num_queries > 0 else 0
    avg_recall = total_recall / num_queries if num_queries > 0 else 0
    avg_mrr = total_mrr / num_queries if num_queries > 0 else 0
    
    return {
        "method": "keyword_search",
        "num_queries": num_queries,
        "correct_count": correct_count,
        "avg_precision_at_k": avg_precision,
        "avg_recall_at_k": avg_recall,
        "avg_mrr": avg_mrr,
        "results": results,
    }


def evaluate_vector_search(index: VectorSearch, embedder: Embedder, ground_truth: List[Dict[str, str]], k: int = 5) -> Dict[str, Any]:
    """Evaluate vector search method."""
    results = []
    correct_count = 0
    total_precision = 0
    total_recall = 0
    total_mrr = 0
    
    for idx, item in enumerate(ground_truth):
        question = item["question"]
        correct_filename = item["filename"]
        
        # Embed query and search
        query_vector = embedder.encode(question)
        search_results = index.search(query_vector, num_results=k)
        
        # Calculate metrics
        metrics = calculate_metrics(search_results, correct_filename, k=k)
        
        results.append({
            "question": question,
            "correct_filename": correct_filename,
            "metrics": metrics,
            "top_k_filenames": metrics["top_k_filenames"]
        })
        
        correct_count += metrics["is_relevant"]
        total_precision += metrics["precision_at_k"]
        total_recall += metrics["recall_at_k"]
        total_mrr += metrics["mrr"]
    
    num_queries = len(ground_truth)
    avg_precision = total_precision / num_queries if num_queries > 0 else 0
    avg_recall = total_recall / num_queries if num_queries > 0 else 0
    avg_mrr = total_mrr / num_queries if num_queries > 0 else 0
    
    return {
        "method": "vector_search",
        "num_queries": num_queries,
        "correct_count": correct_count,
        "avg_precision_at_k": avg_precision,
        "avg_recall_at_k": avg_recall,
        "avg_mrr": avg_mrr,
        "results": results,
    }


def evaluate_hybrid_search(text_index: Index, vector_index: VectorSearch, embedder: Embedder, 
                          ground_truth: List[Dict[str, str]], k: int = 5) -> Dict[str, Any]:
    """Evaluate hybrid search method (RRF combination)."""
    results = []
    correct_count = 0
    total_precision = 0
    total_recall = 0
    total_mrr = 0
    
    for idx, item in enumerate(ground_truth):
        question = item["question"]
        correct_filename = item["filename"]
        
        # Get text search results
        text_results = text_index.search(question, num_results=k)
        
        # Get vector search results
        query_vector = embedder.encode(question)
        vector_results = vector_index.search(query_vector, num_results=k)
        
        # Combine using RRF
        hybrid_results = rrf([text_results, vector_results], num_results=k)
        
        # Calculate metrics
        metrics = calculate_metrics(hybrid_results, correct_filename, k=k)
        
        results.append({
            "question": question,
            "correct_filename": correct_filename,
            "metrics": metrics,
            "top_k_filenames": metrics["top_k_filenames"]
        })
        
        correct_count += metrics["is_relevant"]
        total_precision += metrics["precision_at_k"]
        total_recall += metrics["recall_at_k"]
        total_mrr += metrics["mrr"]
    
    num_queries = len(ground_truth)
    avg_precision = total_precision / num_queries if num_queries > 0 else 0
    avg_recall = total_recall / num_queries if num_queries > 0 else 0
    avg_mrr = total_mrr / num_queries if num_queries > 0 else 0
    
    return {
        "method": "hybrid_search",
        "num_queries": num_queries,
        "correct_count": correct_count,
        "avg_precision_at_k": avg_precision,
        "avg_recall_at_k": avg_recall,
        "avg_mrr": avg_mrr,
        "results": results,
    }


def main():
    import os
    from pathlib import Path
    
    print("Loading ground-truth Q&A pairs...")
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} ground-truth Q&A pairs")
    
    print("\nLoading lesson documents...")
    documents = load_lesson_documents()
    print(f"Loaded {len(documents)} lesson documents")
    
    print("\nBuilding keyword search index...")
    keyword_index = build_keyword_index(documents)
    
    print("\nEvaluating keyword search...")
    keyword_eval = evaluate_keyword_search(keyword_index, ground_truth, k=5)
    print(f"Keyword Search - Avg Precision@5: {keyword_eval['avg_precision_at_k']:.4f}")
    print(f"Keyword Search - Avg Recall@5: {keyword_eval['avg_recall_at_k']:.4f}")
    print(f"Keyword Search - Avg MRR@5: {keyword_eval['avg_mrr']:.4f}")
    print(f"Keyword Search - Correct in top-5: {keyword_eval['correct_count']}/{keyword_eval['num_queries']}")
    
    print("\nPreparing vector search (chunking documents and embedding)...")
    embedder = Embedder(path="../models/Xenova/all-MiniLM-L6-v2")
    chunks = chunk_documents(documents, size=2000, step=1000)
    print(f"Created {len(chunks)} chunks")
    
    chunk_texts = [chunk["content"] for chunk in chunks]
    print("Encoding chunks...")
    chunk_vectors = embedder.encode_batch(chunk_texts)
    print(f"Encoded {len(chunk_vectors)} chunk vectors")
    
    print("\nBuilding vector search index...")
    vector_index = build_vector_index(chunks, chunk_vectors)
    
    print("\nBuilding text search index (for hybrid)...")
    text_index = build_text_index(chunks)
    
    print("\nEvaluating vector search...")
    vector_eval = evaluate_vector_search(vector_index, embedder, ground_truth, k=5)
    print(f"Vector Search - Avg Precision@5: {vector_eval['avg_precision_at_k']:.4f}")
    print(f"Vector Search - Avg Recall@5: {vector_eval['avg_recall_at_k']:.4f}")
    print(f"Vector Search - Avg MRR@5: {vector_eval['avg_mrr']:.4f}")
    print(f"Vector Search - Correct in top-5: {vector_eval['correct_count']}/{vector_eval['num_queries']}")
    
    print("\nEvaluating hybrid search (RRF)...")
    hybrid_eval = evaluate_hybrid_search(text_index, vector_index, embedder, ground_truth, k=5)
    print(f"Hybrid Search - Avg Precision@5: {hybrid_eval['avg_precision_at_k']:.4f}")
    print(f"Hybrid Search - Avg Recall@5: {hybrid_eval['avg_recall_at_k']:.4f}")
    print(f"Hybrid Search - Avg MRR@5: {hybrid_eval['avg_mrr']:.4f}")
    print(f"Hybrid Search - Correct in top-5: {hybrid_eval['correct_count']}/{hybrid_eval['num_queries']}")
    
    # Create comprehensive answers markdown
    answers_md = f"""# Homework 4: Search Evaluation — Answers

## Summary

Evaluated three search methods on {len(ground_truth)} ground-truth Q&A pairs using metrics: Precision@5, Recall@5, and MRR@5.

| Method | Precision@5 | Recall@5 | MRR@5 | Correct Count |
|--------|------------|----------|-------|---------------|
| Keyword Search | {keyword_eval['avg_precision_at_k']:.4f} | {keyword_eval['avg_recall_at_k']:.4f} | {keyword_eval['avg_mrr']:.4f} | {keyword_eval['correct_count']}/{keyword_eval['num_queries']} |
| Vector Search | {vector_eval['avg_precision_at_k']:.4f} | {vector_eval['avg_recall_at_k']:.4f} | {vector_eval['avg_mrr']:.4f} | {vector_eval['correct_count']}/{vector_eval['num_queries']} |
| Hybrid Search (RRF) | {hybrid_eval['avg_precision_at_k']:.4f} | {hybrid_eval['avg_recall_at_k']:.4f} | {hybrid_eval['avg_mrr']:.4f} | {hybrid_eval['correct_count']}/{hybrid_eval['num_queries']} |

## Q1. Keyword Search Evaluation

**Answer:** Precision@5 = {keyword_eval['avg_precision_at_k']:.4f}, Recall@5 = {keyword_eval['avg_recall_at_k']:.4f}, MRR = {keyword_eval['avg_mrr']:.4f}

### Explanation

Keyword search was evaluated using exact text matching on the lesson document content field. The index was built with `minsearch` using:
- Text fields: `content` (full lesson text)
- Keyword fields: `filename` (for grouping)

For each of the {len(ground_truth)} ground-truth questions:
1. Query was sent to the keyword index
2. Top-5 results were retrieved
3. Checked if the correct filename appeared in top-5
4. Calculated precision = (# relevant in top-5) / 5
5. Calculated recall = (# relevant in top-5) / (# total relevant) = is_relevant (since only 1 correct answer per query)
6. Calculated MRR = 1 / rank of first relevant document

Results show that keyword search correctly identified the relevant lesson in **{keyword_eval['correct_count']} out of {keyword_eval['num_queries']}** queries when using top-5 results. This corresponds to an average precision of {keyword_eval['avg_precision_at_k']:.1%}, meaning on average 1 out of 5 results is correct.

### Sample Results (First 5 Queries)

"""
    
    for i, result in enumerate(keyword_eval['results'][:5]):
        answers_md += f"\n**Query {i+1}:** {result['question'][:80]}...\n"
        answers_md += f"- Correct filename: {result['correct_filename']}\n"
        answers_md += f"- Top-5 results: {', '.join(result['top_k_filenames'])}\n"
        answers_md += f"- Found: {'✓ Yes' if result['metrics']['is_relevant'] else '✗ No'}\n"
    
    answers_md += f"""

## Q2. Vector Search Evaluation

**Answer:** Precision@5 = {vector_eval['avg_precision_at_k']:.4f}, Recall@5 = {vector_eval['avg_recall_at_k']:.4f}, MRR = {vector_eval['avg_mrr']:.4f}

### Explanation

Vector search was evaluated using semantic similarity:
1. Documents were chunked into {len(chunks)} chunks (size=2000, step=1000) to better capture topic-specific content
2. Each chunk's content was embedded using the `all-MiniLM-L6-v2` model
3. Queries were embedded using the same model
4. Similarity was measured using dot product (cosine similarity on normalized embeddings)
5. Top-5 most similar chunks were retrieved for each query

Results show that vector search correctly identified relevant content in **{vector_eval['correct_count']} out of {vector_eval['num_queries']}** queries. Average precision of {vector_eval['avg_precision_at_k']:.1%} indicates semantic similarity effectively captures question-content relationships.

## Q3. Hybrid Search Evaluation (RRF)

**Answer:** Precision@5 = {hybrid_eval['avg_precision_at_k']:.4f}, Recall@5 = {hybrid_eval['avg_recall_at_k']:.4f}, MRR = {hybrid_eval['avg_mrr']:.4f}

### Explanation

Hybrid search combines keyword and vector search results using Reciprocal Rank Fusion (RRF):
- Both keyword and vector search return their top-5 results
- Results are combined using RRF formula: score(d) = Σ 1/(k + rank) for each ranking
- Final top-5 results are selected based on combined scores
- k parameter = 60 (standard RRF constant)

Results show **{hybrid_eval['correct_count']} out of {hybrid_eval['num_queries']}** queries had correct results in top-5, with average precision of {hybrid_eval['avg_precision_at_k']:.1%}. Hybrid approach often performs better than individual methods by combining complementary strengths.

## Comparison and Recommendations

### Best Performer
The **{'Keyword Search' if keyword_eval['avg_precision_at_k'] >= max(vector_eval['avg_precision_at_k'], hybrid_eval['avg_precision_at_k']) else ('Vector Search' if vector_eval['avg_precision_at_k'] >= hybrid_eval['avg_precision_at_k'] else 'Hybrid Search')}** method achieved the highest precision@5 with {max(keyword_eval['avg_precision_at_k'], vector_eval['avg_precision_at_k'], hybrid_eval['avg_precision_at_k']):.4f}.

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

- **Ground-truth dataset:** {len(ground_truth)} Q&A pairs
- **Lesson documents:** {len(documents)} documents
- **Chunks:** {len(chunks)} chunks (size=2000, step=1000)
- **Embedding model:** Xenova/all-MiniLM-L6-v2 (local ONNX)
- **Evaluation metric:** Precision@5, Recall@5, MRR@5
- **Dataset source:** DataTalksClub/llm-zoomcamp (commit 8c1834d)
"""

    return keyword_eval, vector_eval, hybrid_eval, answers_md


if __name__ == "__main__":
    keyword_eval, vector_eval, hybrid_eval, answers_md = main()
    
    # Save answers to file
    with open("answers.md", "w", encoding="utf-8") as f:
        f.write(answers_md)
    
    print("\n" + "="*60)
    print("Answers saved to answers.md")
    print("="*60)
