from __future__ import annotations
from typing import Any, Callable
import uuid
from src.backend.agents.chunker import Chunk

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

class EmbeddingStore:
    def __init__(
        self,
        collection_name: str = "student_documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None

        try:
            import chromadb
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, chunk: Chunk) -> dict[str, Any]:
        return {
            "id": chunk.id or str(uuid.uuid4()),
            "content": chunk.text,
            "embedding": self._embedding_fn(chunk.text),
            "metadata": chunk.metadata or {}
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records:
            return []
            
        query_embedding = self._embedding_fn(query)
        
        scored_records = []
        for rec in records:
            score = _dot(query_embedding, rec["embedding"])
            scored_records.append({**rec, "score": score})
        
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, chunks: list[Chunk], progress_callback=None) -> None:
        """Embed each document's content and store it."""
        ids = [chunk.id or str(uuid.uuid4()) for chunk in chunks]
        contents = [chunk.text for chunk in chunks]
        
        metadatas = []
        for chunk in chunks:
            meta = chunk.metadata.copy() if chunk.metadata else {}
            if "doc_id" not in meta and chunk.id:
                meta["doc_id"] = chunk.id
            metadatas.append(meta)

        # Batch encoding if available
        batch_fn = getattr(self._embedding_fn, "encode_batch", None)
        if batch_fn and len(chunks) > 1:
            embeddings = batch_fn(contents)
        else:
            embeddings = []
            for i, chunk in enumerate(chunks):
                embeddings.append(self._embedding_fn(chunk.text))
                if progress_callback:
                    progress_callback(i + 1, len(chunks))

        if self._use_chroma:
            self._collection.add(
                ids=ids,
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            for i in range(len(ids)):
                self._store.append({
                    "id": ids[i],
                    "content": contents[i],
                    "embedding": embeddings[i],
                    "metadata": metadatas[i]
                })

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
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

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k,
                where=metadata_filter
            )
            return [{
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            } for i in range(len(results['ids'][0]))]
        else:
            filtered_records = self._store
            if metadata_filter:
                filtered_records = [
                    rec for rec in self._store 
                    if all(rec["metadata"].get(k) == v for k, v in metadata_filter.items())
                ]
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.
        """
        try:
            if self._use_chroma:
                existing = self._collection.get(
                    where={"doc_id": doc_id},
                    include=[]
                )
                
                if not existing or not existing.get('ids'):
                    return False
                
                self._collection.delete(
                    where={"doc_id": doc_id}
                )
                return True
            else:
                initial_count = len(self._store)
                self._store = [
                    res for res in self._store 
                    if res["metadata"].get("doc_id") != doc_id
                ]
                return len(self._store) < initial_count
                
        except Exception as e:
            return False