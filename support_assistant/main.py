"""
Support Assistant - FastAPI wrapper
================================================================
Run locally with: uvicorn main:app --host 0.0.0.0 --port 7860

MOCK_LLM defaults to mock mode (the graded baseline) if left unset.
"""

from fastapi import FastAPI
from schemas import AskRequest, AskResponse
from graph import ask

app = FastAPI(title="Zepto Support Assistant")


@app.get("/")
def root():
    return {"status": "ok", "service": "zepto-support-assistant"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    return ask(request.query)
