# Module 3 — Support Assistant (`/support_assistant`)

A small, complete RAG service for Zepto: 8 policy documents, embedded
locally and indexed in ChromaDB, orchestrated through a LangGraph flow that
routes each query, retrieves grounded context, and returns a
Pydantic-validated JSON answer via a FastAPI `/ask` endpoint.

**Graded baseline is fully offline.** With `MOCK_LLM` left unset (or `=1`),
no LLM call is made anywhere — intent classification is a keyword
heuristic and answer generation is deterministic/templated. Only
embedding (`sentence-transformers`) and vector storage (`ChromaDB`) run for
real, and both are local, free, and keyless.

## Files

| File | Purpose |
|---|---|
| `docs/doc_01.txt` … `doc_08.txt` | The 8 Zepto policy documents (verbatim, as specified). |
| `ingest.py` | Loads, chunks (one chunk per document), embeds with `all-MiniLM-L6-v2`, and stores vectors in a persistent ChromaDB collection at `chroma_store/`. |
| `rag.py` | Query embedding + top-k retrieval from that collection. |
| `prompt_template.py` | The role-context-task-format-length structured prompt, with a negative constraint and a few-shot example, used by the optional real-LLM extension. |
| `graph.py` | The LangGraph `StateGraph`: `classify_intent` → (`retrieve_and_answer` \| `direct_answer`), the `MOCK_LLM` branching, and the `SupportAnswer` Pydantic schema. |
| `main.py` | FastAPI app exposing `POST /ask`. |
| `Dockerfile` | Builds and serves the FastAPI app locally on port 7860. |

## How to run

```bash
pip install -r requirements.txt
python ingest.py                 # builds the ChromaDB index (needs internet
                                  # on first run, to download the embedding
                                  # model from huggingface.co — one-time,
                                  # no account, no API key)
uvicorn main:app --host 0.0.0.0 --port 7860   # MOCK_LLM defaults to mock mode
```

Docker:

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

> **Sandbox note:** this development container has no network route to
> `huggingface.co` (only a short allow-list of package registries is
> reachable), so the embedding-model download itself could not be executed
> here. All ingestion/retrieval/graph/FastAPI logic was validated
> end-to-end in this environment using a temporary, test-only deterministic
> stand-in for `SentenceTransformer.encode()` (not part of this repo) —
> confirming chunking, ChromaDB storage/query, the LangGraph routing, the
> mock-mode answer templates, and the FastAPI request/response cycle all
> work correctly. With normal internet access, `python ingest.py` downloads
> the real model once and everything below runs unchanged.

## Example calls (MOCK_LLM at its default)

These transcripts were captured against the real graph/FastAPI code above
(via `TestClient`, with the retrieval step running against a stand-in
embedder for the reason noted above — request/response shape and routing
are otherwise exactly what a live `uvicorn` server returns).

**Call 1 — triggers retrieval (`policy_question`):**
```
POST /ask
{"query": "Can I cancel my order after it's packed?"}

200 OK
{
  "answer": "Based on the retrieved context: Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer be...",
  "sources": ["doc_05", "doc_04", "doc_06"],
  "confidence": 1.0
}
```

**Call 2 — does not trigger retrieval (`general_question`):**
```
POST /ask
{"query": "Tell me a joke"}

200 OK
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## RAG pipeline architecture

**Ingestion.** `ingest.py` reads all 8 files under `docs/`. Because each
document is short and already single-topic (one policy area per file), it
is chunked at the document level — one chunk per file, no further
splitting.

**Embedding.** Each chunk's text is embedded locally with
`sentence-transformers`' `all-MiniLM-L6-v2` model (`ingest.py`,
`rag.py::_get_model`). No API key, no network call beyond the one-time
model weight download.

**Storage.** Vectors are written to a persistent ChromaDB collection named
`zepto_policies` at `chroma_store/` (`ingest.py::build_index`).

**Retrieval.** `rag.py::retrieve_top_k` embeds the incoming query with the
same model and pulls the top-3 nearest chunks from that ChromaDB
collection by cosine distance. This is called from the `retrieve_and_answer`
node in `graph.py`, and it runs identically in both `MOCK_LLM` states —
retrieval never depends on the toggle.

**Routing.** `graph.py::classify_intent` decides `policy_question` vs.
`general_question`; `graph.py::route_from_intent` is the LangGraph
conditional edge that sends the query to `retrieve_and_answer` or
`direct_answer` accordingly. This routing logic is also independent of
`MOCK_LLM`.

**Generation — where `MOCK_LLM` branches.** Inside both
`retrieve_and_answer` and `direct_answer`, only the final answer-generation
step checks `MOCK_LLM`:
- **Default / `MOCK_LLM=1` (graded baseline):** `retrieve_and_answer`
  returns a canned `f"Based on the retrieved context: {top_chunk_snippet}"`
  string; `direct_answer` returns a fixed "I can only answer questions
  about Zepto policies right now." string. No LLM is called in either case,
  and the `SupportAnswer` schema is populated deterministically in code
  (`sources` = retrieved chunk ids for policy questions, empty otherwise;
  `confidence` = a fixed `1.0`).
- **`MOCK_LLM=0` (optional, ungraded extension):** `retrieve_and_answer`
  builds the structured prompt from `prompt_template.py` around the
  retrieved chunks and prompts a real LLM (stubbed in `graph.py::_call_llm`,
  intended to be wired to Groq's free tier or another free-tier provider);
  `direct_answer` prompts the LLM directly with no retrieved context. Both
  paths validate the LLM's raw JSON output against the `SupportAnswer`
  Pydantic model and retry up to twice with a corrective instruction before
  returning a clearly marked error response (`graph.py::_llm_generate_validated`).

**Output contract.** Every path — mock or real-LLM — ultimately returns a
`SupportAnswer` (`answer: str`, `sources: list[str]`, `confidence: float`),
which `main.py`'s `POST /ask` endpoint returns as the validated response
model.
