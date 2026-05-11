import streamlit as st
from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod
from src.backend.agents.prompt import SYSTEM_PROMPT, ROUTER_PROMPT
from typing import Any
load_dotenv()

class BaseAssistant(ABC):
    @abstractmethod
    def get_response(self, chat_history: list[dict[str, str]]) -> str:
        """"""

    @abstractmethod
    def get_router_response(self, query: str) -> str:
        """"""

class OpenAIAssistant(BaseAssistant):
    def __init__(self, model: str, api_key: str):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(
            api_key=api_key
        )

    def get_response(self, chat_history: list[dict[str, str]], context: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(CONTEXT=context)}, *chat_history]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return response.choices[0].message.content
    
    def get_router_response(self, query: str) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": query}
            ]
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

        contents = [f"{his["role"]}: {his["content"]}" for his in chat_history]

        response = self.client.models.generate_content(
            model="gemma-4-31b-it",
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="high"),
                system_instruction=SYSTEM_PROMPT.format(CONTEXT=context)
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


