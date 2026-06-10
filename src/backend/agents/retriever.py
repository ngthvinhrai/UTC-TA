import os

from src.backend.core.schema import SearchResult
from src.backend.store.store import KnowledgeStore

DEFAULT_K = os.getenv("DEFAULT_K")

class Retriever:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        result = self.store.bm25_search(query, top_k)
        return self.store.vector_search(query, top_k)
    
    def _fusion(
        self,
        bm25_results: list[SearchResult],
        vector_results: list[SearchResult]
    ) -> list[SearchResult]:
        return _rrf(
            bm25_results=bm25_results,
            vector_results=vector_results
        )
    
    def _rerank(
        self,
        fusion_result: list[SearchResult]
    ) -> list[SearchResult]:
        pass

def _rrf(
        bm25_results: list[SearchResult],
        vector_results: list[SearchResult]
) -> list[SearchResult]:

    bm25_chunk_id = [result.chunk.metadata.chunk_id for result in bm25_results]
    vector_chunk_id = [result.chunk.metadata.chunk_id for result in vector_results]

    intersect = []
    non_intersect_b = []
    non_intersect_v = []

    for i, a in enumerate(bm25_chunk_id):
        if a in vector_chunk_id:
            j = vector_chunk_id.index(a)
            intersect.append((i, j))
        else:
            non_intersect_b.append(i)
    for j, b in enumerate(vector_chunk_id):
        if b not in bm25_chunk_id:
            non_intersect_v.append(j)
    
    search_results = []
    for bi, vi in intersect:
        b_result = bm25_results[bi]
        v_result = vector_results[vi]
        score = 1/(DEFAULT_K + b_result.rank) + 1/(DEFAULT_K + v_result.rank)
        search_results.append((b_result.chunk, score))
    for bi in non_intersect_b:
        b_result = bm25_results[bi]
        score = 1/(DEFAULT_K + b_result.rank)
        search_results.append((b_result.chunk, score))
    for vi in non_intersect_v:
        v_result = vector_results[vi]
        score = 1/(DEFAULT_K + v_result.rank)
        search_results.append((v_result.chunk, score))

    search_results = sorted(search_results, key=lambda x: x[1], reverse=True)

    return [
        SearchResult(
            chunk=chunk,
            score=score,
            rank=i+1,
            method="fusion"
        )
        for i, (chunk, score) in enumerate(search_results)
    ]