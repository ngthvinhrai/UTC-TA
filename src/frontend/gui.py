from src.backend.processing import file_converter
from src.backend.graph import get_graph
from logs.logging import result_logging, chat_logging
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
import os, time, json, random
import markdown
import re

load_dotenv()

def _protect_math(text: str):
    """Replace LaTeX math blocks with placeholders so markdown doesn't mangle them."""
    math_blocks = []

    def replace_display(match):
        math_blocks.append(match.group(0))
        return f"%%MATH_BLOCK_{len(math_blocks) - 1}%%"

    def replace_inline(match):
        math_blocks.append(match.group(0))
        return f"%%MATH_INLINE_{len(math_blocks) - 1}%%"

    text = re.sub(r'\$\$[\s\S]*?\$\$', replace_display, text)
    text = re.sub(r'(?<!\$)\$(?!\$)([^\n$]+)\$(?!\$)', replace_inline, text)
    text = re.sub(r'\\\[[\s\S]*?\\\]', replace_display, text)
    text = re.sub(r'\\\([^\)]*?\\\)', replace_inline, text)

    return text, math_blocks


def _restore_math(html: str, math_blocks: list):
    """Restore protected math blocks into the HTML."""
    for i, block in enumerate(math_blocks):
        if block.startswith('$$') or block.startswith('\\['):
            html = html.replace(f"%%MATH_BLOCK_{i}%%", block)
        elif block.startswith('$') or block.startswith('\\('):
            html = html.replace(f"%%MATH_INLINE_{i}%%", block)
    return html


def markdown_with_math(text: str) -> str:
    """Convert markdown to HTML while preserving LaTeX math delimiters."""
    protected, math_blocks = _protect_math(text)
    html = markdown.markdown(protected, extensions=['extra', 'codehilite'])
    return _restore_math(html, math_blocks)


def render_chat(role, content):
    """Render all chat messages in a single iframe with MathJax."""
    alignment_class = "user-row" if role == "user" else "assistant-row"
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    avatar_icon = "👤" if role == "user" else "🤖"
    content_html = markdown_with_math(content)
    message_htmls = f"""
    <div class="message-row {alignment_class}">
        <div class="avatar">{avatar_icon}</div>
        <div class="chat-bubble {bubble_class}">
            {content_html}
        </div>
    </div>"""

    html_page = f"""<!DOCTYPE html>
<html>
<head>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
<style>
    body {{ margin: 0; padding: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
    .message-row {{ display: flex; width: 100%; margin-bottom: 15px; gap: 12px; }}
    .user-row {{ flex-direction: row-reverse; }}
    .assistant-row {{ flex-direction: row; }}
    .chat-bubble {{ 
        padding: 12px 18px; border-radius: 18px; max-width: 75%; 
        word-wrap: break-word; font-size: 15px; line-height: 1.6; 
        width: fit-content; 
    }}
    .chat-bubble p {{ margin: 0 0 8px 0; text-align: justify; }}
    .chat-bubble p:last-child {{ margin-bottom: 0; }}
    .chat-bubble pre {{ background-color: #272822; color: white; padding: 10px; border-radius: 8px; overflow-x: auto; }}
    .chat-bubble code {{ font-family: 'Courier New', monospace; font-size: 14px; }}
    .user-bubble {{ background-color: #007bff; color: white; border-bottom-right-radius: 2px; }}
    .assistant-bubble {{ background-color: #f0f2f6; color: black; border-bottom-left-radius: 2px; }}
    .avatar {{ font-size: 24px; padding-top: 5px; }}
    mjx-container {{ overflow-x: auto; overflow-y: hidden; max-width: 100%; }}
    mjx-container[display] {{ margin: 0.5em 0; }}
</style>
</head>
<body>
{message_htmls}
</body>
</html>"""
    return html_page
    components.html(html_page, height=200, scrolling=True)


class Interface:
    def __init__(self):
        st.set_page_config(page_title="UTC Teaching Assistant", page_icon="🧮", layout="wide")
        
        if "has_retrieved" not in st.session_state:
            st.session_state.has_retrieved = False
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "Open AI"

        self.initialize()

    def initialize(self):
        with st.sidebar:
            st.title("📚 Tài liệu bổ trợ")
            uploaded_files = st.file_uploader("Upload file", type=["pdf", "txt", "docx"], accept_multiple_files=True)
            if st.button("Nạp tài liệu"):
                if uploaded_files:
                    from src.backend.embedding import get_embedder
                    from src.backend.agents.chunker import chunk_structure_aware
                    from src.backend.processing import file_converter, convert_single_file
                    from src.backend.store.embedding_store import EmbeddingStore
                    from src.backend.agents.retriever import SemanticSearch

                    progress_bar = st.progress(0, text="Đang chuyển đổi tài liệu...")
                    converted_files = file_converter(uploaded_files)
                    progress_bar.progress(100, text="Hoàn tất chuyển đổi!")

                    progress_bar.progress(0, text="Đang chia nhỏ tài liệu...")
                    combined_text = "\n".join(converted_files)
                    chunks = chunk_structure_aware(combined_text)
                    progress_bar.progress(100, text=f"Đã chia thành {len(chunks)} đoạn!")

                    progress_bar.progress(0, text="Đang tạo embedding...")
                    embedder = get_embedder()
                    store = EmbeddingStore(embedding_fn=embedder)

                    def embed_progress(current, total):
                        progress_bar.progress(current / total, text=f"Đang embedding: {current}/{total}")

                    store.add_documents(chunks=chunks, progress_callback=embed_progress)
                    progress_bar.progress(100, text="Hoàn tất embedding!")

                    st.session_state.retriever = SemanticSearch(store)
                    st.session_state.has_retrieved = True
                    progress_bar.empty()
                    st.success(f"Đã nạp tài liệu thành công! ({len(chunks)} đoạn)")
                else: 
                    st.error("Chưa có tài liệu")

            model_options = ["Open AI", "Gemma"]            
            selected_model = st.selectbox(
                "",
                options=model_options,
                index=model_options.index(st.session_state.selected_model)
            )

        st.markdown("### 🧮 UTC Teaching Assistant")

        chat_history: list[dict[str, str]] = []
        for message in st.session_state.messages:
            st.html(render_chat(message["role"], message["content"]), unsafe_allow_javascript=True)
            

        assistant = get_graph(llm=selected_model)

        if query := st.chat_input("Bạn muốn hỏi gì..."):
            st.session_state.messages.append({"role": "user", "content": query})
            st.html(render_chat("user", query), unsafe_allow_javascript=True)
            chat_history = st.session_state.messages.copy()[-7:]

            full_response = ""
            message_placeholder = st.empty()
            waiting_text = random.choice(["Đang suy nghĩ...", "Không có việc gì làm à mà cứ le ve ở đây thế", "Đợi tí!!"])
            message_placeholder.html(
                render_chat("assistant", f"*{waiting_text}*"),
                unsafe_allow_javascript=True 
            )
  
            initial_state = {
                "chat_history": chat_history,
                "context": "",
                "route": "",
                "response": ""
            }

            result = assistant.app.invoke(initial_state)
            response = result["response"]

            for chunk in response.split(" "):
                if chunk is not None: full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.html(
                    render_chat("assistant", full_response + ""), 
                    unsafe_allow_javascript=True
                )
            
            message_placeholder.empty()
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            chat_logging(
                chat=[{"role": "user", "content": query}, {"role": "assistant", "content": full_response}],
                route=result.get("route", ""),
                context=result.get("context", "")
            )
            st.rerun()
