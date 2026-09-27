# Support Assistant — Module 3

See the root `README.md` for setup/run instructions and design decisions.

## Example call transcripts (MOCK_LLM left at default)

Run locally via `uvicorn main:app --host 0.0.0.0 --port 7860`.

**Call 1 — should trigger retrieval (`policy_question`):**
```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "How long do I have to return a damaged item?"}'
```
Actual response:
```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect...",
  "sources": ["doc_02", "doc_06", "doc_05"],
  "confidence": 1.0
}
```

**Call 2 — should NOT trigger retrieval (`general_question`):**
```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
```
Actual response:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

Both calls were run locally with `MOCK_LLM` left at its default (unset), confirming the keyword-based intent classifier correctly routes a policy question to `retrieve_and_answer` (with real ChromaDB retrieval against the `all-MiniLM-L6-v2` embeddings) and an unrelated question to `direct_answer`, with no LLM/network call made in either case.