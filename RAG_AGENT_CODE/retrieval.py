from __future__ import annotations

import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class Retriever:
    """Loads the FAISS index once and returns the top-k chunks for a query."""

    def __init__(self, embeddings, index_path: str = "./faiss_index", top_k: int = 4):
        self.index_path = index_path
        self.top_k = top_k
        self._retriever = self._load(embeddings)

    def _load(self, embeddings):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(
                f"FAISS index not found at '{self.index_path}'. "
                "Build it first with:  python ingest.py ./docs"
            )
        store = FAISS.load_local(
            self.index_path,
            embeddings,
            # Safe here: the index is one we built ourselves, locally.
            allow_dangerous_deserialization=True,
        )
        return store.as_retriever(search_kwargs={"k": self.top_k})

    def retrieve_docs(self, query: str) -> list[Document]:
        if not query:
            return []
        return self._retriever.invoke(query)

    def retrieve(self, query: str) -> str:
        """Return the top-k chunks joined into a single context string."""
        docs = self.retrieve_docs(query)
        return "\n\n".join(d.page_content for d in docs)


class WebSearchTool:
    """Optional web-search fallback via Tavily. Silently no-ops if disabled."""

    def __init__(
        self,
        enabled: bool = False,
        max_results: int = 3,
        api_key: str | None = None,
    ):
        self._tool = None
        if enabled:
            try:
                from langchain_community.tools import TavilySearchResults

                # Pass the key explicitly: it lives in Settings (loaded from
                # .env by pydantic), not in os.environ, so the wrapper can't
                # auto-discover it.
                self._tool = TavilySearchResults(
                    max_results=max_results, tavily_api_key=api_key
                )
            except Exception as exc:
                print(f"[retrieval] Tavily unavailable: {exc}")

    @property
    def available(self) -> bool:
        return self._tool is not None

    def search(self, query: str) -> str:
        if not self.available or not query:
            return ""
        try:
            results = self._tool.invoke(query)
        except Exception as exc:
            print(f"[retrieval] Web search failed: {exc}")
            return ""
        return "\n\n".join(
            f"[Web] {r.get('title', '')}: {r.get('content', '')}" for r in results
        )
