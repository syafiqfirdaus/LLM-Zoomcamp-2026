import json
import os
import time
from search import load_kb_data, MinSearchEngine, VectorSearchEngine

def run_ingestion_pipeline():
    print("[Ingestion Pipeline] Starting automated dataset ingestion...")
    start_time = time.time()
    
    docs = load_kb_data()
    print(f"[Ingestion Pipeline] Loaded {len(docs)} quantitative finance documents.")
    
    # 1. Initialize BM25 Index
    print("[Ingestion Pipeline] Building MinSearch BM25 keyword index...")
    bm25 = MinSearchEngine(docs)
    
    # 2. Build Vector Embeddings
    print("[Ingestion Pipeline] Generating vector embeddings using MiniLM...")
    vec = VectorSearchEngine(docs)
    
    elapsed = time.time() - start_time
    print(f"[Ingestion Pipeline] Ingestion completed successfully in {elapsed:.2f} seconds.")
    print(f"[Ingestion Pipeline] Vector embedding shape: {vec.embeddings.shape}")
    return len(docs)

if __name__ == "__main__":
    run_ingestion_pipeline()
