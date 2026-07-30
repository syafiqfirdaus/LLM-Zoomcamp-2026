import json
import os
import math
import numpy as np
from typing import List, Dict, Any, Tuple
import minsearch
from sentence_transformers import SentenceTransformer

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "quant_finance_kb.json")

class QueryRewriter:
    """Rewrites/expands quantitative finance user queries for improved retrieval recall."""
    SYNONYMS = {
        "bs": "Black-Scholes European call put options pricing model",
        "var": "Value at Risk VaR historical parametric Monte Carlo loss",
        "cvar": "Expected Shortfall Conditional Value at Risk CVaR tail loss",
        "mdd": "Maximum Drawdown peak to trough equity decline",
        "hft": "High-Frequency Trading market making microsecond latency order book",
        "greeks": "Delta Gamma Vega Theta Rho sensitivity hedging",
        "iv": "Implied Volatility Volatility Smile Skew Surface",
        "pe": "Price to Earnings ratio P/E fundamental valuation",
        "statarb": "Statistical Arbitrage pairs trading mean reversion cointegration",
        "capm": "Capital Asset Pricing Model Beta systematic risk expected return"
    }

    @classmethod
    def rewrite(cls, query: str) -> str:
        words = query.lower().split()
        expanded = [query]
        for w in words:
            clean_w = w.strip("?,.!")
            if clean_w in cls.SYNONYMS:
                expanded.append(cls.SYNONYMS[clean_w])
        return " ".join(expanded)


class MinSearchEngine:
    """BM25 Keyword Search Engine powered by minsearch."""
    def __init__(self, docs: List[Dict[str, Any]]):
        # minsearch/sklearn text vectorizers expect plain strings, so normalize
        # keyword lists into a searchable string representation first.
        self.docs = [self._normalize_doc(doc) for doc in docs]
        self.index = minsearch.Index(
            text_fields=["title", "content", "category", "keywords"],
            keyword_fields=["id", "category"]
        )
        self.index.fit(self.docs)

    @staticmethod
    def _normalize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(doc)
        keywords = normalized.get("keywords", [])
        if isinstance(keywords, list):
            normalized["keywords"] = " ".join(str(keyword) for keyword in keywords)
        return normalized

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.index.search(
            query=query,
            filter_dict={},
            boost_dict={"title": 2.0, "keywords": 1.5, "content": 1.0},
            num_results=top_k
        )
        return results


class VectorSearchEngine:
    """Dense Vector Search Engine using SentenceTransformers (all-MiniLM-L6-v2)."""
    def __init__(self, docs: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2"):
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        
        # Prepare text representation for each document
        self.doc_texts = [
            f"{doc['title']} - {doc['category']}: {doc['content']} Keywords: {' '.join(doc.get('keywords', []))}"
            for doc in docs
        ]
        self.embeddings = self.model.encode(self.doc_texts, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        # Cosine similarity (since normalized, dot product = cosine sim)
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_copy = dict(self.docs[idx])
            doc_copy["score"] = float(similarities[idx])
            results.append(doc_copy)
        return results


class HybridSearchEngine:
    """Hybrid Search Engine combining MinSearch (BM25) and Vector Search with Reciprocal Rank Fusion (RRF)."""
    def __init__(self, bm25_engine: MinSearchEngine, vector_engine: VectorSearchEngine):
        self.bm25_engine = bm25_engine
        self.vector_engine = vector_engine

    def search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        bm25_results = self.bm25_engine.search(query, top_k=top_k * 2)
        vector_results = self.vector_engine.search(query, top_k=top_k * 2)

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        # Process BM25 ranks
        for rank, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Process Vector ranks
        for rank, doc in enumerate(vector_results):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort by combined RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        results = []
        for doc_id in sorted_doc_ids:
            doc_copy = dict(doc_map[doc_id])
            doc_copy["rrf_score"] = float(rrf_scores[doc_id])
            results.append(doc_copy)
        return results


def rerank_documents(query: str, docs: List[Dict[str, Any]], vector_engine: VectorSearchEngine) -> List[Dict[str, Any]]:
    """Re-ranks retrieved candidates using vector dot similarity with the query."""
    if not docs:
        return []
    
    doc_texts = [f"{d['title']} {d['content']}" for d in docs]
    query_emb = vector_engine.model.encode([query], normalize_embeddings=True)[0]
    doc_embs = vector_engine.model.encode(doc_texts, normalize_embeddings=True)
    
    scores = np.dot(doc_embs, query_emb)
    scored_docs = []
    for doc, score in zip(docs, scores):
        d_copy = dict(doc)
        d_copy["rerank_score"] = float(score)
        scored_docs.append(d_copy)
        
    scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_docs


# Load data and initialize engines
def load_kb_data() -> List[Dict[str, Any]]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    docs = load_kb_data()
    bm25 = MinSearchEngine(docs)
    vec = VectorSearchEngine(docs)
    hybrid = HybridSearchEngine(bm25, vec)
    
    test_q = "How do I calculate Delta and option price sensitivity?"
    print(f"Query: {test_q}")
    print("\n--- BM25 Results ---")
    for r in bm25.search(test_q, top_k=2):
        print(f"[{r['id']}] {r['title']}")
        
    print("\n--- Vector Results ---")
    for r in vec.search(test_q, top_k=2):
        print(f"[{r['id']}] {r['title']} (score: {r['score']:.4f})")
        
    print("\n--- Hybrid Results ---")
    for r in hybrid.search(test_q, top_k=2):
        print(f"[{r['id']}] {r['title']} (RRF: {r['rrf_score']:.4f})")
