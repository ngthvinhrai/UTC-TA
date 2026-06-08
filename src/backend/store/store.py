from __future__ import annotations
import os
import requests
from typing import Any
import uuid
from src.backend.core.schema import Chunk, SearchResult
from rank_bm25 import BM25Okapi
from turbovec.langchain import TurboQuantVectorStore

FASTAPI_URL = os.getenv("FASTAPI_URL")

class KnowledgeStore:
    def __init__(
        self,
        chunks: list[Chunk],
        # embedding: Any
    ) -> "KnowledgeStore":
        self._chunks = chunks
        # self.embedding = get_embedder()
        self.bm25_index = self._build_bm25_index(chunks)
        self.vector_index = self._build_vector_index(chunks)

    def _build_bm25_index(self, chunks: list[Chunk]) -> BM25Okapi:
        corpus = [self._tokenize(chunk.text) for chunk in chunks]
        store = BM25Okapi(corpus=corpus) 
        return store
    
    def _build_vector_index(self, chunks: list[Chunk]) -> Any:
        chunk_list = [chunk.text for chunk in chunks]
        metadatas = [{
            "id": chunk.id,
            "metadata": chunk.metadata
        } for chunk in chunks]

        resposne = requests.post(
            url=f"{FASTAPI_URL}/vector_store",
            json={
                "text": chunk_list,
                "metadata": metadatas
            }
        )

        return resposne.json()["store"]
    
    def bm25_search(self, query: str, top_k: int=10) -> list[SearchResult]:
        if top_k <= 0 or not self._chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25_index.get_scores(query_tokens)
        top_indexes = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[
            :top_k
        ]

        results = []
        for rank, chunk_index in enumerate(top_indexes, start=1):
            score = float(scores[chunk_index])
            results.append(
                SearchResult(
                    chunk=self._chunks[chunk_index],
                    score=score,
                    rank=rank,
                    method="bm25",
                )
            )

        return results
    
    def vector_search(self, query: str, top_k: int=10) -> list[SearchResult]:
        response = requests.post(
            url=f"{FASTAPI_URL}/vector_search",
            json={
                "query": query,
                "top_k": top_k
            }
        )

        search_result = response.json()

        result = []

        for i, (doc, score) in enumerate(search_result):
            result.append(
                SearchResult(
                    chunk=Chunk(
                        text=doc["page_content"],
                        id=doc["metadata"]["id"],
                        metadata=doc["metadata"]["metadata"]
                    ),
                    score=score,
                    rank=i + 1,
                    method="vector",
                )
            )

        return result
    
    def _tokenize(self, text: str) -> list[str]:
        return text.split()
    
def get_embedder():

    if os.getenv("EMBEDDING_MODEL"):
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL")
        )

    from langchain_openai.embeddings.base import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

