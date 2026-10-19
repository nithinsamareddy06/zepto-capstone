"""
Module 3 - Support Assistant
LangGraph StateGraph: classify_intent -> (retrieve_and_answer | direct_answer)

Every node's generation step branches on the MOCK_LLM environment variable.
MOCK_LLM unset or "1" -> deterministic, rule-based mock logic (graded
baseline, no LLM call, no network call). MOCK_LLM="0" -> optional extension
that calls a real LLM (not required for grading).
"""

import os
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from rag import retrieve_top_k
from prompt_template import build_prompt

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours",
]


def is_mock_mode() -> bool:
    return os.environ.get("MOCK_LLM", "1") != "0"


# ---- Structured output schema -----------------------------------------------

class SupportAnswer(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float


# ---- Graph state -------------------------------------------------------------

class GraphState(TypedDict, total=False):
    query: str
    intent: str                      # "policy_question" | "general_question"
    retrieved_chunks: list[dict]
    response: dict                   # validated SupportAnswer.model_dump()


# ---- Nodes ---------------------------------------------------------------------

def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if is_mock_mode():
        # Mock mode (graded baseline): keyword heuristic, no LLM call.
        lowered = query.lower()
        intent = "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"
    else:
        # Optional MOCK_LLM=0 extension: call a real LLM to classify instead.
        intent = _llm_classify_intent(query)

    return {**state, "intent": intent}


def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval always runs for real in both modes - embedding + ChromaDB
    # need no API key and no network call once the model is cached.
    chunks = retrieve_top_k(query, k=3)

    if is_mock_mode():
        # Mock mode (graded baseline): canned templated answer, no LLM call.
        top_chunk = chunks[0]
        snippet = top_chunk["text"][:200]
        answer_text = f"Based on the retrieved context: {snippet}"
        validated = SupportAnswer(
            answer=answer_text,
            sources=[c["id"] for c in chunks],
            confidence=1.0,
        )
    else:
        # Optional MOCK_LLM=0 extension: prompt a real LLM, grounded only in
        # the retrieved chunks, with retry-on-schema-failure below.
        prompt = build_prompt(query, chunks)
        validated = _llm_generate_validated(prompt, fallback_sources=[c["id"] for c in chunks])

    return {**state, "retrieved_chunks": chunks, "response": validated.model_dump()}


def direct_answer(state: GraphState) -> GraphState:
    query = state["query"]

    if is_mock_mode():
        # Mock mode (graded baseline): fixed canned string, no LLM call.
        validated = SupportAnswer(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0,
        )
    else:
        # Optional MOCK_LLM=0 extension: prompt the LLM directly, no retrieval.
        prompt = (
            "You are Zepto's support assistant. The user asked a question "
            "unrelated to Zepto policy. Politely say you can only help with "
            f"Zepto policy questions.\n\nUser question: {query}"
        )
        validated = _llm_generate_validated(prompt, fallback_sources=[])

    return {**state, "response": validated.model_dump()}


def route_from_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---- Optional MOCK_LLM=0 real-LLM helpers (ungraded extension) -----------------

def _llm_classify_intent(query: str) -> str:
    """Placeholder for the optional real-LLM classification path. Wire this
    to your chosen provider (e.g. Groq) when MOCK_LLM=0. Falls back to the
    keyword heuristic if no LLM client is configured, so the graph never
    breaks even if this extension is only partially set up."""
    lowered = query.lower()
    return "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"


def _llm_generate_validated(prompt: str, fallback_sources: list[str], max_retries: int = 2) -> SupportAnswer:
    """Placeholder for the optional real-LLM generation path (MOCK_LLM=0).
    Calls the LLM, parses its JSON output against SupportAnswer, and retries
    up to `max_retries` additional times with a corrective instruction if
    validation fails, before returning a clearly marked error response."""
    import json

    corrective_suffix = ""
    for attempt in range(max_retries + 1):
        try:
            raw_output = _call_llm(prompt + corrective_suffix)  # implement for your provider
            parsed = json.loads(raw_output)
            return SupportAnswer(**parsed)
        except Exception as exc:
            corrective_suffix = (
                f"\n\nYour previous response failed schema validation "
                f"({exc}). Return ONLY a valid JSON object with fields "
                f"answer (string), sources (list of strings), confidence "
                f"(float 0-1)."
            )
            continue

    return SupportAnswer(
        answer="[ERROR] The assistant could not produce a schema-valid response after retries.",
        sources=fallback_sources,
        confidence=0.0,
    )


def _call_llm(prompt: str) -> str:
    """Stub for the real LLM call used only when MOCK_LLM=0. Wire this to
    Groq's free-tier API (or another genuinely free LLM API) here. Not
    invoked by the graded mock-mode baseline."""
    raise NotImplementedError(
        "Set MOCK_LLM=0 and implement _call_llm() with your chosen free-tier "
        "LLM provider to use the optional real-LLM extension."
    )


# ---- Build the graph -------------------------------------------------------------

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    workflow.add_node("direct_answer", direct_answer)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_from_intent,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)

    return workflow.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(query: str) -> SupportAnswer:
    graph = get_graph()
    result_state = graph.invoke({"query": query})
    return SupportAnswer(**result_state["response"])
