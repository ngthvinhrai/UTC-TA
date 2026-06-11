from __future__ import annotations
import os
import requests
from src.backend.core.schema import Chunk, SearchResult
from rank_bm25 import BM25Okapi

FASTAPI_URL = os.getenv("FASTAPI_URL")

class KnowledgeStore:
    def __init__(
        self,
        chunks: list[Chunk],
        previous_chunks: list[Chunk] | None = None
    ) -> "KnowledgeStore":
        self._chunks = chunks + previous_chunks if previous_chunks else chunks
        self._previous_chunks = previous_chunks
        self.bm25_index = self._build_bm25_index(self._chunks)
        self._build_vector_index(chunks)

    def _build_bm25_index(self, chunks: list[Chunk]) -> BM25Okapi:
        corpus = [self._tokenize(chunk.text) for chunk in chunks]
        store = BM25Okapi(corpus=corpus) 
        return store
    
    def _build_vector_index(self, chunks: list[Chunk]):
        response = requests.get(
            url=f"{FASTAPI_URL}/vector_store_status"
        )
        if response.json()["code"] == 404 or self._previous_chunks is not None:
            chunk_list = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [chunk.id for chunk in chunks]

            for i in range(0, len(chunk_list), 250):
                end_idx = min(i + 250, len(chunk_list))    
                requests.post(
                    url=f"{FASTAPI_URL}/vector_store",
                    json={
                        "text": chunk_list[i:end_idx],
                        "metadata": metadatas[i:end_idx],
                        "id": ids[i:end_idx]
                    }
                )
    
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
            print(doc)
            result.append(
                SearchResult(
                    chunk=Chunk(
                        text=doc["page_content"],
                        id=doc["id"],
                        metadata=doc["metadata"]
                    ),
                    score=score,
                    rank=i + 1,
                    method="vector",
                )
            )

        return result
    
    def _tokenize(self, text: str) -> list[str]:
        return text.split()
    
    def save(self, path: str) -> None:
        import json
        
        if not os.path.exists(path): 
            os.makedirs(path, exist_ok=True)
        
        with open(path + "/chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self._chunks:
                json.dump(chunk.model_dump(), f, ensure_ascii=False)
                f.write("\n")

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

