from __future__ import annotations
from typing import Any
import uuid
from src.backend.core.schema import Chunk, SearchResult
from rank_bm25 import BM25Okapi
from turbovec.langchain import TurboQuantVectorStore

class KnowledgeStore:
    def __init__(
        self,
        chunks: list[Chunk],
        # embedding: Any
    ) -> "KnowledgeStore":
        self._chunks = chunks
        self.embedding = get_embedder()
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

        store = TurboQuantVectorStore(embedding=self.embedding).from_texts(
            texts=chunk_list,
            embedding=self.embedding,
            metadatas=metadatas
        )

        return store
    
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
        search_result = self.vector_index.similarity_search_with_score(
            query=query,
            k=top_k
        )

        result = []

        for i, (doc, score) in enumerate(search_result):
            result.append(
                SearchResult(
                    chunk=Chunk(
                        text=doc.page_content,
                        id=doc.metadata["id"],
                        metadata=doc.metadata["metadata"]
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
    from langchain_openai.embeddings.base import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
    )