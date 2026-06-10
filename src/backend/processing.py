import tempfile
import os, re
import streamlit as st
import pymupdf4llm
from markitdown import MarkItDown
import networkx as nx
from underthesea import word_tokenize, pos_tag


def _convert_pdf(file_path: str, progress_bar=None) -> str:
    import pymupdf
    doc = pymupdf.open(file_path)
    total_pages = len(doc)

    md_pages = []
    for i in range(total_pages):
        md_page = pymupdf4llm.to_markdown(
            doc,
            pages=[i],
            table_strategy="lines",
            use_ocr=False,
            ignore_images=False,
            ignore_graphics=False,
            page_chunks=True,
        )
        if isinstance(md_page, list):
            md_pages.extend(md_page)
        else:
            md_pages.append(md_page)

        if progress_bar:
            progress_bar.progress((i + 1) / total_pages, text=f"Đang xử lý trang {i + 1}/{total_pages}")

    if not md_pages:
        return ""

    if isinstance(md_pages[0], dict):
        return "\n\n".join(chunk.get("markdown", chunk.get("text", "")) for chunk in md_pages if chunk.get("markdown") or chunk.get("text"))
    return "\n\n".join(str(p) for p in md_pages)


def _convert_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _convert_docx(file_path: str) -> str:
    converter = MarkItDown(enable_plugins=False)
    return converter.convert(file_path).markdown


def file_converter(uploaded_files: list) -> list:
    combined_md = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

            if ext == "pdf":
                progress_bar = st.progress(0, text=f"Đang chuyển đổi: {uploaded_file.name}")
                md_content = _convert_pdf(file_path, progress_bar)
                progress_bar.empty()
            elif ext == "txt":
                md_content = _convert_txt(file_path)
            elif ext in ("docx", "doc"):
                md_content = _convert_docx(file_path)
            else:
                md_content = ""

            combined_md.append(md_content)

    return combined_md


def convert_single_file(file_path: str) -> str:
    ext = file_path.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return _convert_pdf(file_path)
    elif ext == "txt":
        return _convert_txt(file_path)
    elif ext in ("docx", "doc"):
        return _convert_docx(file_path)
    return ""


def tokenize(text):
    # underthesea works for both VI + EN

    stopwords = {
        "và", "là", "của", "có", "trong",
        "một", "những", "được", "với",
        "khi", "đó", "này", "cho",
        "the", "a", "an", "is", "are",
        "was", "were", "of", "to",
        "in", "on", "for", "and"
    }

    # tokens = word_tokenize(text)
    tagged = pos_tag(text)
    candidates = []
    for word, pos in tagged:
        word = word.lower()
        if (
            word not in stopwords
            and re.match(r"^[\w_]+$", word)
            and pos in ["N", "Np", "A", "V"]
        ):
            candidates.append(word)
    return candidates
# ========================================
# BUILD GRAPH
# ========================================
def build_graph(words, window_size=4):
    graph = nx.Graph()
    for i in range(len(words)):
        for j in range(
            i + 1,
            min(i + window_size, len(words))
        ):
            w1 = words[i]
            w2 = words[j]
            if w1 != w2:
                if graph.has_edge(w1, w2):
                    graph[w1][w2]["weight"] += 1
                else:
                    graph.add_edge(
                        w1,
                        w2,
                        weight=1
                    )
    return graph
# ========================================
# TEXTRANK
# ========================================
def textrank(graph):
    return nx.pagerank(
        graph,
        weight="weight"
    )
# ========================================
# EXTRACT KEYPHRASES
# ========================================
def extract_keyphrase(text, top_k=10):
    words = tokenize(text)
    if not words:
        return []
    graph = build_graph(words)
    scores = textrank(graph)
    ranked_vocab = {
        word: score
        for word, score in scores.items()
    }
    # Original token sequence
    original_tokens = word_tokenize(text)
    phrases = []
    current_phrase = []
    for token in original_tokens:
        normalized = token.lower()
        if normalized in ranked_vocab:
            current_phrase.append(normalized)
        else:
            if current_phrase:
                phrase = " ".join(current_phrase)
                score = (
                    sum(ranked_vocab[w] for w in current_phrase)
                    / len(current_phrase)
                )
                phrases.append((phrase, score))
                current_phrase = []
    # Last phrase
    if current_phrase:
        phrase = " ".join(current_phrase)
        score = (
            sum(ranked_vocab[w] for w in current_phrase)
            / len(current_phrase)
        )
        phrases.append((phrase, score))
    # Remove duplicates
    unique_phrases = {}
    for phrase, score in phrases:
        if phrase not in unique_phrases:
            unique_phrases[phrase] = score
    ranked_phrases = sorted(
        unique_phrases.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked_phrases[:top_k]

