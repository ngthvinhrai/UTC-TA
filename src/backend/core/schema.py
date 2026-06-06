from typing import Any, Callable
from typing import TypedDict
from pydantic import BaseModel
from dataclasses import field

class Chunk(BaseModel):
    text: str
    metadata: dict = field(default_factory=dict)
    id: str | None = ""
    parent_id: str | None = None

class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    method: str

class GraphState(TypedDict):
    chat_history: list[dict[str, str]]
    context: list[SearchResult]
    route: str
    response: str


