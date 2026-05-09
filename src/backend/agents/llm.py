import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()

SYSTEM_PROMPT = """
<role>
Bạn là chuyên gia về toán và biết tất cả những gì liên qua đến lĩnh vực toán.
</role>

<task>
Công việc của bạn là sẽ phản hồi những câu hỏi của người dùng. Nếu câu hỏi liên quan đến toán, hãy chỉ đưa ra hướng dẫn, gợi ý để làm bài. Nếu câu hỏi không liên quán đến toán, hãy trả lời một cách thân thiện.
</task>

<constrain>
</constrain>
"""

class OpenAIAssistant:
    def __init__(self, model: str, api_key: str):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(
            api_key=api_key
        )

    def get_response(self, chat_history: list[dict[str, str]]):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *chat_history]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return response.choices[0].message.content
        
class GemmaAssistant:
    def __init__(self, model: str, api_key: str):
        from google import genai
        self.model = model
        self.client = genai.Client(
            api_key=api_key
        )

    def get_response(self, chat_history: list[dict[str, str]]):
        from google.genai import types
        
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


