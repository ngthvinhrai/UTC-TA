from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from src.backend.agents.llm import get_assistant
from logs.logging import context_logging, chat_logging
import streamlit as st

class GraphState(TypedDict):
    chat_history: list[dict[str, str]]
    context: str = ""
    route: str
    response: str

class Graph:
    def __init__(self, llm):
        self.llm = llm
        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)

        graph.add_node("router", self.route_node)
        graph.add_node("retriever", self.retrieve_node)
        graph.add_node("generator", self.generate_node)

        graph.add_conditional_edges(
            "router",
            lambda x: x["route"],
            {
                "retrieve": "retriever",
                "direct": "generator"
            }
        )

        graph.add_edge("retriever", "generator")
        graph.add_edge("generator", END)
        graph.set_entry_point("router") 

        return graph.compile()
    
    def route_node(self, state: GraphState) -> dict:
        if not st.session_state.get("has_retrieved", False): return {"route": "direct"}
        # chat_logging(state["chat_history"])


        decision = self.llm.get_router_response(state["chat_history"][-1]["content"])

        if "retrieve" in decision.strip().lower():
            return {"route": "retrieve"}
        else:
            return {"route": "direct"}

    def retrieve_node(self, state: GraphState) -> dict:
        contexts = st.session_state.retriever.search(state["chat_history"][-1]["content"])
        # context_logging(contexts)
        context = "\n".join([c["content"] for c in contexts])

        return {"context": context}

    def generate_node(self, state: GraphState) -> dict:
        response = self.llm.get_response(
            chat_history=state["chat_history"],
            context=state["context"]
        )

        return {"response": response}

@st.cache_resource
def get_graph(llm):
    return Graph(
        llm=get_assistant(llm),
    )