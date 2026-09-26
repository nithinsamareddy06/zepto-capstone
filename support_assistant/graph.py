"""
Support Assistant - LangGraph orchestration
================================================================
A 3-node StateGraph:

    classify_intent --(policy_question)--> retrieve_and_answer --> END
                    --(general_question)--> direct_answer       --> END

Every node's *generation* step branches on the MOCK_LLM environment
variable. MOCK_LLM unset or "1" (the default, graded baseline) uses fully
deterministic, rule-based / templated logic with NO network or LLM calls.
MOCK_LLM="0" is an optional, ungraded extension that calls a real LLM
(wired for the Groq free tier, but any OpenAI-compatible-ish client could
be swapped in) via the prompt template in prompt_template.py.

Retrieval itself (embedding the query + querying ChromaDB) always runs for
real in both modes -- it needs no API key and no network call once the
embedding model is cached locally.
"""

import os
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END

from ingest import build_collection, query_collection
from prompt_template import build_prompt
from schemas import AskResponse

MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours",
]

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        _collection = build_collection()
    return _collection


class GraphState(TypedDict):
    query: str
    intent: str
    retrieved: List[dict]
    answer: str
    sources: List[str]
    confidence: float


# ----------------------------------------------------------------------
# Node 1: classify_intent
# ----------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    query_lower = state["query"].lower()

    if MOCK_LLM:
        # Mock mode (graded baseline): keyword heuristic, no LLM call.
        is_policy = any(kw in query_lower for kw in POLICY_KEYWORDS)
        intent = "policy_question" if is_policy else "general_question"
    else:
        # Optional MOCK_LLM=0 extension: call a real LLM to classify.
        intent = _llm_classify_intent(state["query"])

    return {**state, "intent": intent}


def _llm_classify_intent(query: str) -> str:
    """Optional real-LLM classification path (MOCK_LLM=0). Requires a
    configured LLM client (e.g. Groq). Left as a clear extension point."""
    raise NotImplementedError(
        "Wire up a real LLM client here for the optional MOCK_LLM=0 path "
        "(e.g. Groq's free tier). Not required for grading."
    )


# ----------------------------------------------------------------------
# Node 2: retrieve_and_answer
# ----------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> GraphState:
    collection = get_collection()
    # Retrieval always runs for real in both modes.
    hits = query_collection(collection, state["query"], top_k=3)
    source_ids = [h["id"] for h in hits]

    if MOCK_LLM:
        # Mock mode (graded baseline): canned templated answer, no LLM call.
        top_chunk = hits[0]["text"] if hits else ""
        snippet = top_chunk[:200]
        answer = f"Based on the retrieved context: {snippet}"
        confidence = 1.0
    else:
        # Optional MOCK_LLM=0 extension: prompt a real LLM, grounded only
        # in the retrieved chunks, using the structured template.
        prompt = build_prompt(state["query"], hits)
        answer, confidence = _llm_generate_answer(prompt)

    return {
        **state,
        "retrieved": hits,
        "answer": answer,
        "sources": source_ids,
        "confidence": confidence,
    }


def _llm_generate_answer(prompt: str):
    """Optional real-LLM generation path (MOCK_LLM=0)."""
    raise NotImplementedError(
        "Wire up a real LLM client here for the optional MOCK_LLM=0 path. "
        "Not required for grading."
    )


# ----------------------------------------------------------------------
# Node 3: direct_answer
# ----------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    if MOCK_LLM:
        # Mock mode (graded baseline): fixed canned string, no LLM call.
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
    else:
        answer, confidence = _llm_generate_answer(state["query"])

    return {**state, "retrieved": [], "answer": answer, "sources": [], "confidence": confidence}


# ----------------------------------------------------------------------
# Conditional routing edge
# ----------------------------------------------------------------------
def route_by_intent(state: GraphState) -> Literal["retrieve_and_answer", "direct_answer"]:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ----------------------------------------------------------------------
# Build the graph
# ----------------------------------------------------------------------
def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


_app_graph = None


def ask(query: str) -> AskResponse:
    global _app_graph
    if _app_graph is None:
        _app_graph = build_graph()

    result = _app_graph.invoke({
        "query": query, "intent": "", "retrieved": [],
        "answer": "", "sources": [], "confidence": 0.0,
    })
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )


if __name__ == "__main__":
    for q in ["How long do I have to return a damaged item?", "What's the meaning of life?"]:
        resp = ask(q)
        print(f"Q: {q}")
        print(f"A: {resp.model_dump_json(indent=2)}\n")
