from langgraph.graph import StateGraph, END
from src.backend.agents.llm import get_assistant, BaseAssistant
from src.backend.core.schema import GraphState
from src.backend.agents.retriever import Retriever
import streamlit as st

class Graph:
    def __init__(
        self,
        llm: BaseAssistant,
        retriever: Retriever | None = None
    ) -> "Graph":
        self.llm = llm
        self.retriever = retriever
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
        # return {"route": "retrieve"}
        if not st.session_state.has_retrieved:
            return {"route": "direct"}
        decision = self.llm.get_router_response(state["chat_history"][-1]["content"])

        if "retrieve" in decision.strip().lower():
            return {"route": "retrieve"}
        else:
            return {"route": "direct"}

    def retrieve_node(self, state: GraphState) -> dict:
        contexts = self.retriever.search("\n".join([f"{c["role"]}: {c["content"]}" for c in state["chat_history"]]))

        return {"context": contexts}

    def generate_node(self, state: GraphState) -> dict:
        response = self.llm.get_response(
            chat_history=state["chat_history"],
            context=[c.chunk.text for c in state["context"]]
        )

        return {"response": response}

@st.cache_resource
def get_graph(llm, retriever):
    return Graph(
        llm=get_assistant(llm),
        retriever=retriever
    )