from src.backend.processing import file_converter
from src.backend.agents.llm import get_assistant
from dotenv import load_dotenv
import streamlit as st
import os
import time
import markdown

load_dotenv()

class Interface:
    def __init__(self):
        st.set_page_config(page_title="UTC Teaching Assistant", page_icon="🧮", layout="wide")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        self.initialize()

    def get_message_html(self, role, content):
        """Tạo chuỗi HTML cho tin nhắn để tái sử dụng trong streaming"""
        alignment_class = "user-row" if role == "user" else "assistant-row"
        bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
        avatar_icon = "👤" if role == "user" else "🤖"
        
        # Convert Markdown to HTML
        content_html = markdown.markdown(content, extensions=['extra', 'codehilite'])
        
        return f"""
        <div class="message-row {alignment_class}">
            <div class="avatar">{avatar_icon}</div>
            <div class="chat-bubble {bubble_class}">
                {content_html}
            </div>
        </div>
        """

    def initialize(self):
        # CSS Custom (Giữ nguyên các quy tắc bám lề và co giãn)
        st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body);"></script>
<style>
    .message-row { display: flex; width: 100%; margin-bottom: 15px; gap: 12px; }
    .user-row { flex-direction: row-reverse; }
    .assistant-row { flex-direction: row; }
    .chat-bubble { 
        padding: 12px 18px; border-radius: 18px; max-width: 75%; 
        word-wrap: break-word; font-size: 15px; line-height: 1.6; 
        width: fit-content; 
    }
    .chat-bubble p { margin-bottom: 8px; text-align: justify; }
    .chat-bubble pre { background-color: #272822; color: white; padding: 10px; border-radius: 8px; overflow-x: auto; }
    .user-bubble { background-color: #007bff; color: white; border-bottom-right-radius: 2px; }
    .assistant-bubble { background-color: #f0f2f6; color: black; border-bottom-left-radius: 2px; }
    .avatar { font-size: 24px; padding-top: 5px; }
    [data-testid="stChatMessage"] { display: none !important; }
</style>
        """, unsafe_allow_html=True)

        with st.sidebar:
            st.title("📚 Tài liệu bổ trợ")
            uploaded_files = st.file_uploader("Upload file", type=["pdf", "txt", "docx"], accept_multiple_files=True)
            if st.button("Nạp tài liệu"):
                if uploaded_files:
                    st.success("Đã nạp tài liệu thành công!")
                else: 
                    st.error("Chưa có tài liệu")

        st.markdown("### 🧮 UTC Teaching Assistant")

        chat_history: list[dict[str, str]] = []

        # Hiển thị lịch sử
        for message in st.session_state.messages:
            st.markdown(self.get_message_html(message["role"], message["content"]), unsafe_allow_html=True)

        llm = get_assistant("Gemma")
        
        # Xử lý Input
        if query := st.chat_input("Bạn muốn hỏi gì..."):
            # Hiển thị User Message
            st.session_state.messages.append({"role": "user", "content": query})
            st.markdown(self.get_message_html("user", query), unsafe_allow_html=True)
            chat_history.append({"role": "user", "content": query})

            # Khởi tạo vùng trống cho Assistant streaming
            full_response = ""
            message_placeholder = st.empty()

            # Giả lập streaming (Thay đoạn này bằng generator từ LLM của bạn)
            response = llm.get_response(chat_history=st.session_state.messages)
            
            for chunk in response.split(" "):
                if chunk is not None: full_response += chunk + " "
                time.sleep(0.05) # Tốc độ chạy chữ
                # Cập nhật liên tục HTML vào placeholder
                message_placeholder.markdown(
                    self.get_message_html("assistant", full_response + ""), 
                    unsafe_allow_html=True
                )
            
            # Hoàn tất: Lưu vào history và render bản cuối không có con trỏ ▌
            message_placeholder.markdown(self.get_message_html("assistant", full_response), unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.markdown("<script>renderMathInElement(document.body);</script>", unsafe_allow_html=True)