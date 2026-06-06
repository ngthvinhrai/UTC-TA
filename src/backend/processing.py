import tempfile
import os
import streamlit as st
import pymupdf4llm
from markitdown import MarkItDown


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
