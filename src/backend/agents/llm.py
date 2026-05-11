import streamlit as st
from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod
from typing import Any
load_dotenv()

SYSTEM_PROMPT = """
<role>
Bạn là chuyên gia về toán và biết tất cả những gì liên qua đến lĩnh vực toán.
</role>

<task>
Công việc của bạn là sẽ phản hồi những câu hỏi của người dùng. Nếu câu hỏi liên quan đến toán, hãy chỉ đưa ra hướng dẫn, gợi ý để làm bài. Nếu câu hỏi không liên quán đến toán, hãy trả lời một cách thân thiện.
Bạn sẽ được cung cấp một đoạn tài liệu để trả lời không bị nhầm. Dưới đây là tại liệu đấy:
{CONTEXT}
</task>

<constrain>
</constrain>
"""

class BaseAssistant(ABC):
    @abstractmethod
    def get_response(self, chat_history: list[dict[str, str]]) -> str:
        """"""

    def get_context(self, context: list[dict[str, Any]]) -> str:
        return "\n".join([c["content"] for c in context])

class OpenAIAssistant(BaseAssistant):
    def __init__(self, model: str, api_key: str):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(
            api_key=api_key
        )

    def get_response(self, chat_history: list[dict[str, str]], context: list[dict[str, Any]]) -> str:
        context_content = self.get_context(context)
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(CONTEXT=context_content)}, *chat_history]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return response.choices[0].message.content
        
class GemmaAssistant(BaseAssistant):
    def __init__(self, model: str, api_key: str):
        from google import genai
        self.model = model
        self.client = genai.Client(
            api_key=api_key
        )

    def get_response(self, chat_history: list[dict[str, str]], context: str) -> str:
        from google.genai import types

        context = self.get_context(user_query=chat_history[0]["content"])
        contents = [f"{his["role"]}: {his["content"]}" for his in chat_history]

        response = self.client.models.generate_content(
            model="gemma-4-31b-it",
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="high"),
                system_instruction=SYSTEM_PROMPT
            ),
        )

        return response.text


@st.cache_resource
def get_assistant(model: str):
    match model:
        case "Open AI": 
            return OpenAIAssistant(
                model=os.getenv("OPENAI_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        case "Gemma":
            return GemmaAssistant(
                model=os.getenv("GEMMA_MODEL"),
                api_key=os.getenv("GOOGLE_API_KEY")
            )
        case _:
            return OpenAIAssistant(
                model=os.getenv("OPENAI_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY")
            )


