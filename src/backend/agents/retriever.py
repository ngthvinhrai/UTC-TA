from typing import Any
from dataclasses import dataclass
from dotenv import load_dotenv
import os
load_dotenv()

@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"

class SemanticSearch:
    def __init__(self, store):
        self.store = store

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:

        return self.store.search(query, top_k)
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k
            )
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                })
            return formatted_results
        else:
            return self._search_records(query, self._store, top_k)