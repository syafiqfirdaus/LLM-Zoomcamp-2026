import json
import os
import time
from typing import List, Dict, Any
from search import (
    load_kb_data,
    MinSearchEngine,
    VectorSearchEngine,
    HybridSearchEngine,
    QueryRewriter,
    rerank_documents
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def load_ground_truth() -> List[Dict[str, Any]]:
    gt_path = os.path.join(DATA_DIR, "ground_truth.json")
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_approach(search_fn, ground_truth: List[Dict[str, Any]], k: int = 5) -> Dict[str, float]:
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_k = 0
    reciprocal_ranks = []

    for item in ground_truth:
        query = item["question"]
        expected_doc_id = item["document_id"]
        
        retrieved = search_fn(query, top_k=k)
        retrieved_ids = [doc["id"] for doc in retrieved]

        # Calculate hits
        if expected_doc_id in retrieved_ids[:1]:
            hits_at_1 += 1
        if expected_doc_id in retrieved_ids[:3]:
            hits_at_3 += 1
        if expected_doc_id in retrieved_ids:
            hits_at_k += 1

        # Calculate MRR
        if expected_doc_id in retrieved_ids:
            rank = retrieved_ids.index(expected_doc_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(ground_truth)
    return {
        "hit_rate_@1": round(hits_at_1 / total, 4),
        "hit_rate_@3": round(hits_at_3 / total, 4),
        "hit_rate_@5": round(hits_at_k / total, 4),
        "mrr": round(sum(reciprocal_ranks) / total, 4)
    }

def run_retrieval_evaluation() -> Dict[str, Dict[str, float]]:
    print("Loading Knowledge Base & Ground Truth...")
    docs = load_kb_data()
    ground_truth = load_ground_truth()
    
    print("Initializing Search Engines...")
    bm25 = MinSearchEngine(docs)
    vec = VectorSearchEngine(docs)
    hybrid = HybridSearchEngine(bm25, vec)

    approaches = {
        "MinSearch (BM25)": lambda q, top_k: bm25.search(q, top_k=top_k),
        "Vector Search (MiniLM)": lambda q, top_k: vec.search(q, top_k=top_k),
        "Hybrid Search (RRF)": lambda q, top_k: hybrid.search(q, top_k=top_k),
        "Hybrid + Query Rewriting": lambda q, top_k: hybrid.search(QueryRewriter.rewrite(q), top_k=top_k),
        "Hybrid + Re-ranking": lambda q, top_k: rerank_documents(
            q, hybrid.search(q, top_k=top_k * 2), vec
        )[:top_k]
    }

    results = {}
    print("\n========================================================")
    print("           RETRIEVAL EVALUATION RESULTS TABLE           ")
    print("========================================================")
    header = f"{'Approach':<28} | {'Hit@1':<7} | {'Hit@3':<7} | {'Hit@5':<7} | {'MRR':<7}"
    print(header)
    print("-" * len(header))

    for name, fn in approaches.items():
        metrics = evaluate_approach(fn, ground_truth, k=5)
        results[name] = metrics
        print(f"{name:<28} | {metrics['hit_rate_@1']:<7.4f} | {metrics['hit_rate_@3']:<7.4f} | {metrics['hit_rate_@5']:<7.4f} | {metrics['mrr']:<7.4f}")

    print("========================================================\n")

    # Save results to json for README and UI display
    output_path = os.path.join(DATA_DIR, "retrieval_eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    return results

if __name__ == "__main__":
    run_retrieval_evaluation()
