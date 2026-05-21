"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import re
from dataclasses import dataclass, field

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    id: str | None = ""
    parent_id: str | None = None
# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    
    stat = {
        "Chunks": len(chunks),
        "Avg Len": sum([len(c.text) for c in chunks])/len(chunks),
        "Min": min([len(c.text) for c in chunks]),
        "Max": max([len(c.text) for c in chunks])
    }

    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = 0.8,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    # Implement semantic chunking
    # 1. Split text into sentences:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    #
    # 2. Encode sentences:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)
    #
    # 3. Compare consecutive sentences:
    from numpy import dot
    from numpy.linalg import norm
    def cosine_sim(a, b): return dot(a, b) / (norm(a) * norm(b))
    #
    # 4. Group sentences:
    chunks = []
    current_group = [sentences[0]]
    chunk_index = 0
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i-1], embeddings[i])
        if sim < threshold:
            chunks.append(Chunk(text=" ".join(current_group), metadata={"chunk_index": chunk_index, "strategy": "sementic"}))
            current_group = []
            chunk_index += 1
        current_group.append(sentences[i])
    chunks.append(Chunk(text=" ".join(current_group), metadata={"chunk_index": chunk_index, "strategy": "sementic"}))
    

    # 5. Return chunks with metadata: {"chunk_index": i, "strategy": "semantic"}
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────
def chunk_hierarchical(
    text: str,
    parent_size: int,
    child_size: int,
    metadata: dict | None = None
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Default recommendation cho production RAG.
    """
    metadata = metadata or {}
    parents: list[Chunk] = []
    children: list[Chunk] = []

    # -------- 1. Split text into parent chunks (by paragraph aggregation) --------
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current_parent_parts = []
    current_len = 0
    parent_index = 0

    def flush_parent():
        nonlocal parent_index, current_parent_parts, current_len
        if not current_parent_parts:
            return None

        parent_text = "\n\n".join(current_parent_parts)
        pid = f"parent_{parent_index}"
        parent_chunk = Chunk(
            text=parent_text,
            metadata={
                **metadata,
                "chunk_type": "parent",
                "parent_id": pid,
            },
        )
        parents.append(parent_chunk)

        parent_index += 1
        current_parent_parts = []
        current_len = 0
        return parent_chunk

    for para in paragraphs:
        if current_len + len(para) > parent_size:
            flush_parent()
        current_parent_parts.append(para)
        current_len += len(para)

    last_parent = flush_parent()

    # -------- 2. Split each parent into child chunks (sliding window) --------
    for parent in parents:
        pid = parent.metadata["parent_id"]
        parent_text = parent.text

        start = 0
        text_len = len(parent_text)

        while start < text_len:
            end = start + child_size
            child_text = parent_text[start:end].strip()
            if child_text:
                child_chunk = Chunk(
                    text=child_text,
                    metadata={
                        **metadata,
                        "chunk_type": "child",
                    },
                    parent_id=pid,
                )
                children.append(child_chunk)

            start += child_size  # no overlap; add overlap here if needed

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None, max_chunk_size: int = 1000) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.
        max_chunk_size: Maximum characters per chunk. Sections larger than this
                        will be split by paragraphs to avoid oversized chunks.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    # 1. Split by markdown headers:
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    #
    # 2. Pair headers with their content:
    chunks = []
    current_header = ""
    current_content = ""
    chunk_index = 0

    def finalize_chunk(header: str, content: str):
        nonlocal chunk_index
        combined = f"{header}\n{content}".strip()
        if not combined:
            return
        # If chunk exceeds max_chunk_size, split by paragraphs
        if len(combined) > max_chunk_size:
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            current_part = header + "\n\n" if header else ""
            for para in paragraphs:
                if len(current_part) + len(para) > max_chunk_size and current_part.strip():
                    chunks.append(Chunk(
                        text=current_part.strip(),
                        metadata={**metadata, "section": header, "strategy": "structure", "chunk_index": chunk_index}
                    ))
                    chunk_index += 1
                    current_part = header + "\n\n" if header else ""
                current_part += para + "\n\n"
            if current_part.strip():
                chunks.append(Chunk(
                    text=current_part.strip(),
                    metadata={**metadata, "section": header, "strategy": "structure", "chunk_index": chunk_index}
                ))
                chunk_index += 1
        else:
            chunks.append(Chunk(
                text=combined,
                metadata={**metadata, "section": header, "strategy": "structure", "chunk_index": chunk_index}
            ))
            chunk_index += 1

    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            if current_content.strip():
                finalize_chunk(current_header, current_content)
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    # Final chunk
    finalize_chunk(current_header, current_content)

    return chunks

if __name__ == "__main__":
    pass