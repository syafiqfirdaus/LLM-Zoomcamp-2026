import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:
    from gitsource import GithubRepositoryDataReader, chunk_documents  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "The 'gitsource' package is required. Install it with 'pip install gitsource' "
        "or ensure the package is available on PYTHONPATH."
    ) from exc

from minsearch import Index

try:
    from openai import OpenAI
except ImportError:  # type: ignore
    OpenAI = None

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MODEL = "gpt-5.4-mini"
QUERY = "How does the agentic loop keep calling the model until it stops?"


@dataclass
class RAGResult:
    answer: str
    usage: Dict[str, Any]


def get_openai_client() -> Any:
    api_key = os.environ.get(OPENAI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set {OPENAI_API_KEY_ENV} before running the script")

    if OpenAI is not None:
        return OpenAI(api_key=api_key)

    import openai as openai_lib  # type: ignore
    openai_lib.api_key = api_key
    return openai_lib


def extract_doc(search_item: Any) -> Dict[str, Any]:
    if isinstance(search_item, dict):
        return search_item
    if isinstance(search_item, (list, tuple)) and search_item:
        return extract_doc(search_item[0])
    if hasattr(search_item, "to_dict"):
        return search_item.to_dict()
    if hasattr(search_item, "__dict__"):
        return vars(search_item)
    return {}


def response_to_text(response: Any) -> str:
    if response is None:
        return ""
    if hasattr(response, "output_text"):
        return getattr(response, "output_text")
    if isinstance(response, dict):
        if "output_text" in response:
            return response["output_text"]
        if "choices" in response:
            choices = response["choices"]
            if choices and isinstance(choices, list):
                first = choices[0]
                if isinstance(first, dict):
                    return first.get("message", {}).get("content", "") or first.get("text", "")
    if hasattr(response, "choices"):
        choices = getattr(response, "choices")
        if choices and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                return first.get("message", {}).get("content", "") or first.get("text", "")
    return ""


def response_to_usage(response: Any) -> Dict[str, Any]:
    if response is None:
        return {}
    if hasattr(response, "usage"):
        return getattr(response, "usage") or {}
    if isinstance(response, dict):
        return response.get("usage", {}) or {}
    return {}


class LessonRAG:
    def __init__(self, index: Index, llm_client: Any, model: str = DEFAULT_MODEL):
        self.index = index
        self.llm_client = llm_client
        self.model = model

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results: Iterable[Any]) -> str:
        lines: List[str] = []

        for item in search_results:
            doc = extract_doc(item)
            filename = doc.get("filename", "<unknown>")
            content = doc.get("content", "")
            lines.append(f"FILENAME: {filename}")
            lines.append(content)
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query: str, search_results: Iterable[Any]) -> str:
        context = self.build_context(search_results)
        return (
            "QUESTION:\n"
            f"{query}\n\n"
            "CONTEXT:\n"
            f"{context}"
        )

    def llm(self, prompt: str) -> Any:
        messages = [
            {
                "role": "developer",
                "content": (
                    "Your task is to answer questions from the course participants "
                    "based on the provided context. Use the context to find relevant "
                    "information and provide accurate answers. If the answer is not found "
                    'in the context, respond with "I don\'t know."'
                ),
            },
            {"role": "user", "content": prompt},
        ]

        client = self.llm_client
        if hasattr(client, "responses"):
            return client.responses.create(model=self.model, input=messages)

        if hasattr(client, "ChatCompletion"):
            return client.ChatCompletion.create(model=self.model, messages=messages)

        if hasattr(client, "chat"):
            return client.chat.create(model=self.model, messages=messages)

        raise RuntimeError("Unsupported OpenAI client interface")

    def rag(self, query: str, num_results: int = 5) -> RAGResult:
        results = self.search(query, num_results=num_results)
        prompt = self.build_prompt(query, results)
        response = self.llm(prompt)
        return RAGResult(
            answer=response_to_text(response),
            usage=response_to_usage(response),
        )


def load_lesson_documents() -> List[Dict[str, Any]]:
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    files = reader.read()

    documents = []
    for raw_file in files:
        parsed = raw_file.parse()
        documents.append(
            {
                "filename": parsed["filename"],
                "content": parsed["content"],
            }
        )

    return documents


def build_index(documents: List[Dict[str, Any]]) -> Index:
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(documents)
    return index


def main() -> None:
    documents = load_lesson_documents()
    print(f"Lesson document count: {len(documents)}")

    index = build_index(documents)

    search_results = index.search(QUERY, num_results=5)
    if not search_results:
        raise RuntimeError("No search results were returned")

    first_result = extract_doc(search_results[0])
    top_filename = first_result.get("filename", "<unknown>")
    print(f"Top search result filename: {top_filename}")

    llm_client = get_openai_client()
    rag = LessonRAG(index=index, llm_client=llm_client)

    rag_result = rag.rag(QUERY, num_results=5)
    input_tokens = rag_result.usage.get("input_tokens") or rag_result.usage.get("prompt_tokens")
    print("RAG answer:")
    print(rag_result.answer)
    print("RAG usage metadata:")
    print(rag_result.usage)
    print(f"Input token count: {input_tokens}")

    chunks = chunk_documents(documents, size=2000, step=1000)
    print(f"Chunk count: {len(chunks)}")


if __name__ == "__main__":
    main()