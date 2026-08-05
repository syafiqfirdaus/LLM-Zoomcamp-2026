# Peer Review: `amrmancy/llm-zoomcamp-project-2026`

Repository reviewed: `https://gitlab.com/amrmancy/llm-zoomcamp-project-2026`

This version follows the exact criteria names shown in the screenshots in `Evaluate Project Attempt 1/Review Criteria/`.

## Scores

| Criterion | Score | Reason |
|---|---:|---|
| Retrieval evaluation | 2/2 | Multiple retrieval approaches are evaluated and the best one is used. |
| RAG evaluation | 2/2 | Multiple prompt/RAG approaches are evaluated and one is selected based on results. |
| Interface | 2/2 | Has a Streamlit UI and a FastAPI API. |
| Ingestion pipeline | 2/2 | Automated ingestion is implemented with `dlt` and Python scripts. |
| Monitoring | 2/2 | User feedback is collected and there is a dashboard with at least 5 charts/views. |
| Containerization | 2/2 | Everything is defined in `docker-compose.yml`. |
| Problem description | 2/2 | The problem is well described and the project purpose is clear. |
| RAG flow | 2/2 | Uses both a knowledge base and an LLM in the flow. |
| Reproducibility | 2/2 | Clear run instructions, accessible dataset, and pinned dependencies. |
| Hybrid search | 1/1 | Implemented and evaluated. |
| Document re-ranking | 1/1 | Implemented and evaluated. |
| User query rewriting | 0/1 | Discussed in the README, but not present in the shipped implementation. |
| Deployment to the cloud | 0/2 | No cloud deployment found. |
| Bonus point | 0/1 | No extra feature beyond the listed rubric items that I would separately award here. |
| Bonus point | 0/1 | Not awarded. |
| Bonus point | 0/1 | Not awarded. |

**Total:** `20/26`

## Paste-ready review comment

Strong project overall. The README is very clear, the problem is well framed, and the implementation matches the documentation. I could see solid retrieval evaluation, RAG/prompt evaluation, automated ingestion with `dlt`, a usable Streamlit interface plus FastAPI, monitoring with feedback and dashboard views, and full containerization with `docker-compose`.

The only rubric item I would not award is user query rewriting, because although the README explains that it was explored, I could not find it in the final shipped system. I also did not find a cloud deployment or another separate bonus feature to award extra points for. Aside from that, this is a strong, well-documented submission.
