from typing import Any
from src.backend.core.schema import SearchResult
from src.backend.store.store import KnowledgeStore

class Retriever:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        result = self.store.bm25_search(query, top_k)
        return self.store.vector_search(query, top_k)