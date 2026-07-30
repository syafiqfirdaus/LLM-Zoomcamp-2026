import sqlite3
import json
import time
from typing import List, Dict, Any, Optional
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "quant_pulse.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS query_logs (
        id TEXT PRIMARY KEY,
        timestamp REAL,
        user_query TEXT,
        rewritten_query TEXT,
        search_method TEXT,
        retrieved_doc_ids TEXT,
        answer TEXT,
        model_name TEXT,
        latency_ms REAL,
        prompt_tokens INTEGER,
        completion_tokens INTEGER
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_id TEXT,
        rating INTEGER, -- +1 for helpful, -1 for unhelpful
        comment TEXT,
        timestamp REAL,
        FOREIGN KEY (query_id) REFERENCES query_logs (id)
    )
    """)
    
    conn.commit()
    conn.close()

def save_query_log(
    query_id: str,
    user_query: str,
    rewritten_query: str,
    search_method: str,
    retrieved_doc_ids: List[str],
    answer: str,
    model_name: str,
    latency_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0
):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO query_logs (
            id, timestamp, user_query, rewritten_query, search_method,
            retrieved_doc_ids, answer, model_name, latency_ms, prompt_tokens, completion_tokens
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        query_id,
        time.time(),
        user_query,
        rewritten_query,
        search_method,
        json.dumps(retrieved_doc_ids),
        answer,
        model_name,
        latency_ms,
        prompt_tokens,
        completion_tokens
    ))
    conn.commit()
    conn.close()

def save_user_feedback(query_id: str, rating: int, comment: str = ""):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (query_id, rating, comment, timestamp)
        VALUES (?, ?, ?, ?)
    """, (query_id, rating, comment, time.time()))
    conn.commit()
    conn.close()

def get_recent_logs(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_monitoring_metrics() -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    
    # Total queries
    cursor.execute("SELECT COUNT(*) as total FROM query_logs")
    total_queries = cursor.fetchone()["total"]
    
    # Avg latency
    cursor.execute("SELECT AVG(latency_ms) as avg_latency FROM query_logs")
    avg_latency = cursor.fetchone()["avg_latency"] or 0.0
    
    # Search method breakdown
    cursor.execute("SELECT search_method, COUNT(*) as count FROM query_logs GROUP BY search_method")
    method_counts = {row["search_method"]: row["count"] for row in cursor.fetchall()}
    
    # Model breakdown
    cursor.execute("SELECT model_name, COUNT(*) as count FROM query_logs GROUP BY model_name")
    model_counts = {row["model_name"]: row["count"] for row in cursor.fetchall()}
    
    # Feedback counts
    cursor.execute("SELECT rating, COUNT(*) as count FROM feedback GROUP BY rating")
    feedback_rows = cursor.fetchall()
    feedback_counts = {row["rating"]: row["count"] for row in feedback_rows}
    
    conn.close()
    
    return {
        "total_queries": total_queries,
        "avg_latency_ms": round(avg_latency, 2),
        "method_counts": method_counts,
        "model_counts": model_counts,
        "positive_feedback": feedback_counts.get(1, 0),
        "negative_feedback": feedback_counts.get(-1, 0)
    }

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
