from src.backend.graph import Graph
from src.backend.agents.llm import get_assistant
from src.backend.agents.retriever import Retriever
from src.backend.store.store import KnowledgeStore
from logs.logging import chat_logging, result_logging
from src.backend.core.schema import SearchResult, Chunk
import streamlit as st
import time, random
import markdown
import re
import os

KSTORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "store", "kstore"))

WAITING_TEXTS = [
    "Đang suy nghĩ...",
    "Không có việc gì làm à mà cứ le ve ở đây thế", 
    "Đợi tí!!",
    "FACT: Người phát triển ứng dụng này là 1 trong 500 kỹ sư được VinGroup lựa chọn tham gia chương trình **AI thực chiến**. "
]

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

class Interface:
    def __init__(self):
        st.set_page_config(page_title="UTC Teaching Assistant", page_icon="🧮", layout="wide")
        
        if "has_retrieved" in st.query_params:
            st.session_state.has_retrieved = True
        elif "has_retrieved" not in st.session_state:
            st.session_state.has_retrieved = False
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "OpenAI"

        # self.initialize()

    def initialize(self):
        with st.sidebar:
            st.title("📚 Tài liệu bổ trợ")
            uploaded_files = st.file_uploader("Upload file", type=["pdf", "txt", "docx"], accept_multiple_files=True)
            if st.button("Nạp tài liệu"):
                from src.backend.agents.chunker import chunk_structure_aware
                from src.backend.processing import file_converter
                
                
                # embedder = get_embedder()

                if uploaded_files:

                    progress_bar = st.progress(0, text="Đang chuyển đổi tài liệu...")
                    # progress_bar.progress(100, text="Hoàn tất chuyển đổi!")

                    converted_files = file_converter(uploaded_files)
                    progress_bar.progress(0, text="Đang chia nhỏ tài liệu...")
                    combined_text = "\n".join(converted_files)
                    # with open("test/converted_result.md", "r", encoding="utf-8") as f: combined_text = f.read()
                    chunks = chunk_structure_aware(combined_text)
                    progress_bar.progress(100, text=f"Đã chia thành {len(chunks)} đoạn!")

                    progress_bar.progress(0, text="Đang tạo embedding...")
                    
                    if os.path.exists(KSTORE_DIR):
                        previous_chunks = _load_chunk()
                        store = KnowledgeStore(
                            chunks=chunks,
                            previous_chunks=previous_chunks
                        )
                    else:
                        store = KnowledgeStore(
                            chunks=chunks
                        )
                    st.session_state.store = store
                    store.save(KSTORE_DIR)
                    progress_bar.progress(100, text="Hoàn tất embedding!")

                    st.session_state.has_retrieved = True
                    st.query_params["has_retrieved"] = "true"
                    progress_bar.empty()
                    st.success(f"Đã nạp tài liệu thành công! ({len(chunks)} đoạn)")
                else:
                    st.warning("Vui lòng chọn file để nạp tài liệu.")

            model_options = ["OpenAI", "Gemini", "Gemma", "DeepSeek"]            
            selected_model = st.selectbox(
                "",
                options=model_options,
                index=model_options.index(st.session_state.selected_model)
            )
            st.session_state.selected_model = selected_model

        st.markdown("### 🧮 UTC Teaching Assistant")

        for message in st.session_state.messages:
            st.html(render_chat(message["role"], message["content"]), unsafe_allow_javascript=True)
            
        if "store" in st.session_state:
            retriever = Retriever(st.session_state["store"])
            self.assistant = Graph(
                llm=get_assistant(selected_model),
                retriever=retriever
            )
        elif os.path.exists(KSTORE_DIR):
            retriever = Retriever(
                store=KnowledgeStore(chunks=_load_chunk())
            )
            self.assistant = Graph(
                llm=get_assistant(selected_model),
                retriever=retriever
            )  
        else:
            self.assistant = Graph(
                llm=get_assistant(selected_model),
            )
    
    def run(self):
        if query := st.chat_input("Bạn muốn hỏi gì..."):
            st.session_state.messages.append({"role": "user", "content": query})
            st.html(render_chat("user", query), unsafe_allow_javascript=True)
            chat_history = st.session_state.messages.copy()[-1:]

            full_response = ""
            message_placeholder = st.empty()
            waiting_text = random.choice(WAITING_TEXTS)
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

            result = self.assistant.app.invoke(initial_state)
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
            result_logging(_dump_to_dict(result))
            st.rerun()

def _dump_to_dict(result):
    return {
        "bot_response": result["response"],
        "route": result["route"],
        "contexts": [
            {
                "text": rs.chunk.text,
                "rank": rs.rank,
                "score": rs.score
            }
            for rs in result["context"]
        ]
    }

def _load_chunk():
    import json

    chunks_list = []
    with open(KSTORE_DIR + "/chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunks_list.append(
                Chunk(
                    text=chunk["text"],
                    metadata=chunk["metadata"],
                    id=chunk["id"],
                    parent_id=chunk["parent_id"]
                )
            )
    return chunks_list

