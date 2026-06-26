import numpy as np
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch

try:
    from embedder import Embedder
except ImportError as exc:
    raise ImportError(
        "Missing embedder helper. Place `embedder.py` in `02-vector-search/` "
        "or install the course helper script from the upstream repo."
    ) from exc


def load_lesson_documents():
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


def build_vector_search(chunks, vectors):
    index = VectorSearch(keyword_fields=["filename"])
    index.fit(vectors, chunks)
    return index


def build_text_index(chunks):
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(chunks)
    return index


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def main():
    embedder = Embedder()

    documents = load_lesson_documents()
    print(f"Loaded lesson documents: {len(documents)}")

    chunks = chunk_documents(documents, size=2000, step=1000)
    print(f"Created chunks: {len(chunks)}")

    # Q1
    q1_query = "How does approximate nearest neighbor search work?"
    q1_vector = embedder.encode(q1_query)
    q1_value = float(q1_vector[0])
    print("Q1 answer:")
    print(q1_value)

    # Q2
    page = next(
        doc
        for doc in documents
        if doc["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md"
    )
    page_vector = embedder.encode(page["content"])
    q2_similarity = float(np.dot(page_vector, q1_vector))
    print("Q2 answer:")
    print(q2_similarity)

    # Q3
    chunk_texts = [chunk["content"] for chunk in chunks]
    chunk_vectors = embedder.encode_batch(chunk_texts)
    q3_scores = chunk_vectors.dot(q1_vector)
    top_chunk = chunks[int(np.argmax(q3_scores))]
    q3_filename = top_chunk["filename"]
    print("Q3 answer:")
    print(q3_filename)

    # Q4
    q4_query = "What metric do we use to evaluate a search engine?"
    q4_vector = embedder.encode(q4_query)
    vector_index = build_vector_search(chunks, chunk_vectors)
    vector_results_q4 = vector_index.search(q4_vector, num_results=5)
    q4_filename = vector_results_q4[0]["filename"] if vector_results_q4 else None
    print("Q4 answer:")
    print(q4_filename)

    # Q5
    text_index = build_text_index(chunks)
    text_results_q5 = text_index.search("How do I store vectors in PostgreSQL?", num_results=5)
    vector_results_q5 = vector_index.search(
        embedder.encode("How do I store vectors in PostgreSQL?"), num_results=5
    )
    text_filenames = {doc["filename"] for doc in text_results_q5}
    vector_filenames = [doc["filename"] for doc in vector_results_q5]
    q5_only_vector = next(
        (filename for filename in vector_filenames if filename not in text_filenames),
        None,
    )
    print("Q5 answer:")
    print(q5_only_vector)

    # Q6
    q6_query = "How do I give the model access to tools?"
    q6_vector = embedder.encode(q6_query)
    q6_vector_results = vector_index.search(q6_vector, num_results=5)
    q6_text_results = text_index.search(q6_query, num_results=5)
    q6_results = rrf([q6_vector_results, q6_text_results])
    q6_filename = q6_results[0]["filename"] if q6_results else None
    print("Q6 answer:")
    print(q6_filename)

    print("\nFinal answers:")
    print(f"Q1: {q1_value}")
    print(f"Q2: {q2_similarity}")
    print(f"Q3: {q3_filename}")
    print(f"Q4: {q4_filename}")
    print(f"Q5: {q5_only_vector}")
    print(f"Q6: {q6_filename}")


if __name__ == "__main__":
    main()