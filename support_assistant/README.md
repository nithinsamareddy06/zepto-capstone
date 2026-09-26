# Support Assistant — Module 3

See the root `README.md` for setup/run instructions and design decisions.

## Example call transcripts (MOCK_LLM left at default)

Run the two commands below after starting the server locally
(`uvicorn main:app --host 0.0.0.0 --port 7860`) and paste your actual
output into this file before submitting — this keeps the transcript
authentic to your own local run (embedding retrieval quality depends on
the exact model download/cache on your machine).

**Call 1 — should trigger retrieval (`policy_question`):**
```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "How long do I have to return a damaged item?"}'
```
Expected shape of response:
```json
{
  "answer": "Based on the retrieved context: <top chunk snippet>...",
  "sources": ["doc_06", "..."],
  "confidence": 1.0
}
```

**Call 2 — should NOT trigger retrieval (`general_question`):**
```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
```
Expected response:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

> Verification note: the LangGraph routing, ChromaDB add/query plumbing,
> Pydantic schema validation, and the FastAPI endpoint were all tested
> end-to-end during development. The only piece that could not be
> exercised in the authoring sandbox was the actual download of the
> `all-MiniLM-L6-v2` weights from huggingface.co (blocked by that
> sandbox's outbound network allowlist) — this is a one-time ~90MB
> download that works normally on a regular machine or Colab, after
> which `chroma_db/` caches locally and no further network access is
> needed for embeddings.
