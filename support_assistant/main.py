"""
Module 3 - Support Assistant
FastAPI wrapper around the LangGraph RAG flow.

Run locally: uvicorn main:app --host 0.0.0.0 --port 7860
(MOCK_LLM defaults to mock mode - that's what's graded; export MOCK_LLM=0
only if you've wired up the optional real-LLM extension.)
"""

from fastapi import FastAPI
from pydantic import BaseModel

from graph import ask, SupportAnswer

app = FastAPI(title="Zepto Support Assistant")


class AskRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"status": "ok", "service": "zepto-support-assistant"}


@app.post("/ask", response_model=SupportAnswer)
def ask_endpoint(request: AskRequest):
    return ask(request.query)
