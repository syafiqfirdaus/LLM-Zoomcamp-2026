import streamlit as st
import time
import os
import sys
import uuid
import json
import numpy as np
import scipy.stats as si
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from search import (
    load_kb_data,
    MinSearchEngine,
    VectorSearchEngine,
    HybridSearchEngine,
    QueryRewriter,
    rerank_documents
)
from db import init_db, save_query_log, save_user_feedback, get_monitoring_metrics, get_recent_logs
from evaluate_llm import generate_answer, PROMPT_V2_EXPERT, PROMPT_V1_SIMPLE

# Page Configuration
st.set_page_config(
    page_title="QuantPulse AI - Quantitative Finance Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #546E7A;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database and Engines (Cached)
@st.cache_resource
def initialize_system():
    init_db()
    docs = load_kb_data()
    bm25 = MinSearchEngine(docs)
    vec = VectorSearchEngine(docs)
    hybrid = HybridSearchEngine(bm25, vec)
    return docs, bm25, vec, hybrid

docs, bm25_engine, vec_engine, hybrid_engine = initialize_system()

# Sidebar Configuration
st.sidebar.title("📈 QuantPulse Settings")

model_choice = st.sidebar.selectbox(
    "LLM Model",
    ["gpt-4o-mini", "gpt-4o"],
    index=0
)

retrieval_method = st.sidebar.selectbox(
    "Retrieval Engine",
    ["Hybrid Search (RRF)", "MinSearch (BM25)", "Vector Search (MiniLM)", "Hybrid + Re-ranking"],
    index=0
)

enable_rewrite = st.sidebar.checkbox("Enable Query Rewriting", value=True)
top_k_docs = st.sidebar.slider("Top Documents to Retrieve", min_value=1, max_value=8, value=3)

# App Navigation Tabs
tab1, tab2, tab3 = st.tabs(["💬 Quant Assistant", "📈 Black-Scholes & Market Sandbox", "📊 Monitoring & Evaluation Dashboard"])

# ----------------------------------------------------
# TAB 1: QUANT ASSISTANT
# ----------------------------------------------------
with tab1:
    st.markdown('<div class="main-header">QuantPulse AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">End-to-End Quantitative Finance RAG & Algorithmic Trading Knowledge Engine</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am QuantPulse AI. Ask me anything about options pricing (Black-Scholes, Binomial Trees), risk management (VaR, Sharpe Ratio), algorithmic trading (Pairs Trading, Momentum), or financial fundamentals."}
        ]

    # Render Chat History
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Retrieved Knowledge Base Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"**[{src['id']}] {src['title']}** ({src['category']})")
                        st.caption(src['content'])
            if "query_id" in msg:
                col_fb1, col_fb2, _ = st.columns([1, 1, 8])
                with col_fb1:
                    if st.button("👍", key=f"up_{idx}"):
                        save_user_feedback(msg["query_id"], 1, "Helpful response")
                        st.toast("Thank you for your feedback! (+1)")
                with col_fb2:
                    if st.button("👎", key=f"down_{idx}"):
                        save_user_feedback(msg["query_id"], -1, "Needs improvement")
                        st.toast("Feedback recorded (-1)")

    # User Input
    if user_query := st.chat_input("Ask a quantitative finance question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Process Query
        with st.chat_message("assistant"):
            with st.spinner("Retrieving quantitative documents & generating answer..."):
                start_time = time.time()
                query_id = str(uuid.uuid4())
                
                # 1. Query Rewriting
                effective_query = QueryRewriter.rewrite(user_query) if enable_rewrite else user_query

                # 2. Retrieval
                if retrieval_method == "MinSearch (BM25)":
                    retrieved = bm25_engine.search(effective_query, top_k=top_k_docs)
                elif retrieval_method == "Vector Search (MiniLM)":
                    retrieved = vec_engine.search(effective_query, top_k=top_k_docs)
                elif retrieval_method == "Hybrid + Re-ranking":
                    initial = hybrid_engine.search(effective_query, top_k=top_k_docs * 2)
                    retrieved = rerank_documents(user_query, initial, vec_engine)[:top_k_docs]
                else:  # Hybrid Search (RRF)
                    retrieved = hybrid_engine.search(effective_query, top_k=top_k_docs)

                # 3. LLM Generation
                answer, prompt_tokens, completion_tokens = generate_answer(
                    PROMPT_V2_EXPERT, user_query, retrieved, model=model_choice
                )
                
                latency_ms = (time.time() - start_time) * 1000

                # 4. Save Query Log to SQLite
                doc_ids = [d["id"] for d in retrieved]
                save_query_log(
                    query_id=query_id,
                    user_query=user_query,
                    rewritten_query=effective_query,
                    search_method=retrieval_method,
                    retrieved_doc_ids=doc_ids,
                    answer=answer,
                    model_name=model_choice,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )

                # Display Assistant Answer
                st.markdown(answer)
                
                with st.expander("📚 Retrieved Knowledge Base Sources"):
                    for src in retrieved:
                        st.markdown(f"**[{src['id']}] {src['title']}** ({src['category']})")
                        st.caption(src['content'])
                
                st.caption(f"⚡ Latency: {latency_ms:.1f}ms | Tokens: {prompt_tokens + completion_tokens} | Engine: {retrieval_method}")

                # Save to session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved,
                    "query_id": query_id
                })


# ----------------------------------------------------
# TAB 2: BLACK-SCHOLES & QUANT SANDBOX
# ----------------------------------------------------
with tab2:
    st.header("📈 Interactive Black-Scholes Calculator & Ticker Lookup")
    st.caption("Perform real-time European option valuation and Greeks sensitivity analysis.")

    col_input, col_results = st.columns([1, 1])

    with col_input:
        st.subheader("Option Input Parameters")
        S = st.number_input("Underlying Stock Price (S)", value=100.0, step=1.0)
        K = st.number_input("Strike Price (K)", value=100.0, step=1.0)
        T = st.number_input("Time to Expiration (Years, T)", value=1.0, step=0.1)
        r = st.number_input("Risk-Free Interest Rate (r, e.g. 0.05 = 5%)", value=0.05, step=0.01)
        sigma = st.number_input("Volatility (σ, e.g. 0.20 = 20%)", value=0.20, step=0.01)
        option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)

    with col_results:
        st.subheader("Valuation & Greeks Output")
        
        # Black-Scholes Math
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "Call":
            price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
            delta = si.norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
            delta = si.norm.cdf(d1) - 1.0

        gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * si.norm.pdf(d1) * np.sqrt(T) / 100.0  # scaled per 1% vol change
        theta = (- (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2)) / 365.0

        st.metric("Option Fair Value", f"${price:.4f}")
        
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Delta (Δ)", f"{delta:.4f}")
        g2.metric("Gamma (Γ)", f"{gamma:.4f}")
        g3.metric("Vega (ν)", f"{vega:.4f}")
        g4.metric("Theta (Θ)", f"{theta:.4f}")

    st.markdown("---")
    st.subheader("Live Market Ticker Data Viewer")
    ticker = st.text_input("Enter Stock Ticker Symbol", value="AAPL").upper()
    if st.button("Fetch Market Data"):
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6m")
            if not hist.empty:
                fig = px.line(hist, x=hist.index, y="Close", title=f"{ticker} 6-Month Price History")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No historical data found for ticker {ticker}.")
        except Exception as e:
            st.error(f"Error fetching ticker data: {str(e)}")


# ----------------------------------------------------
# TAB 3: MONITORING & EVALUATION DASHBOARD
# ----------------------------------------------------
with tab3:
    st.header("📊 Application Analytics & Monitoring Dashboard")
    st.caption("Live operational metrics, feedback analytics, and retrieval evaluation benchmarks.")

    metrics = get_monitoring_metrics()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total User Queries", metrics["total_queries"])
    m2.metric("Average Latency", f"{metrics['avg_latency_ms']} ms")
    m3.metric("Helpful Feedback (+1)", metrics["positive_feedback"])
    m4.metric("Unhelpful Feedback (-1)", metrics["negative_feedback"])

    st.markdown("---")

    # Load Retrieval Evaluation Benchmarks
    eval_file = os.path.join(os.path.dirname(__file__), "data", "retrieval_eval_results.json")
    if os.path.exists(eval_file):
        st.subheader("Retrieval Engine Evaluation Benchmarks (Ground Truth Dataset)")
        with open(eval_file, "r") as f:
            eval_data = json.load(f)

        df_eval = pd.DataFrame.from_dict(eval_data, orient="index")
        st.dataframe(df_eval.style.highlight_max(axis=0, color="#C8E6C9"), use_container_width=True)

        # Bar chart for MRR
        fig_mrr = px.bar(
            df_eval,
            y=df_eval.index,
            x="mrr",
            orientation="h",
            title="Mean Reciprocal Rank (MRR) by Search Engine",
            color="mrr",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_mrr, use_container_width=True)

    # Charts for Live Telemetry
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if metrics["method_counts"]:
            fig_methods = px.pie(
                names=list(metrics["method_counts"].keys()),
                values=list(metrics["method_counts"].values()),
                title="Query Breakdown by Search Engine"
            )
            st.plotly_chart(fig_methods, use_container_width=True)
        else:
            st.info("No query logs recorded yet for search breakdown.")

    with col_c2:
        logs = get_recent_logs(limit=30)
        if logs:
            df_logs = pd.DataFrame(logs)
            fig_lat = px.scatter(
                df_logs,
                x="timestamp",
                y="latency_ms",
                color="search_method",
                title="Response Latency (ms) over Time"
            )
            st.plotly_chart(fig_lat, use_container_width=True)
        else:
            st.info("No query latency logs recorded yet.")
