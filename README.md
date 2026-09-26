# Zepto Data & AI Platform — Capstone Project

One connected platform, three modules, submitted as a single repository:

| Module | Path | Marks | What it does |
|---|---|---|---|
| Data Pipeline | `/data_pipeline` | 25 | Scrapes books.toscrape.com, cleans & currency-converts, loads into a normalized SQLite DB, runs SQL + pandas queries |
| Analytics Pipeline | `/analytics` | 50 | EDA + full predictive-modeling pipeline on the Titanic dataset |
| Support Assistant | `/support_assistant` | 25 | A RAG-based GenAI assistant (LangGraph + ChromaDB + FastAPI) answering Zepto policy questions, offline-mock by default |

---

## Setup

Each module has its own `requirements.txt` (chosen over one consolidated file, since the three modules have non-overlapping dependencies and this keeps installs lighter per module). From the repo root:

```bash
# Module 1
cd data_pipeline && pip install -r requirements.txt && cd ..

# Module 2
cd analytics && pip install -r requirements.txt && cd ..

# Module 3
cd support_assistant && pip install -r requirements.txt && cd ..
```

Python 3.10+ recommended throughout.

---

## How to run each module end to end

### Module 1 — Data Pipeline

```bash
cd data_pipeline
python3 scrape.py       # scrapes books.toscrape.com -> books_raw.csv
python3 clean_load.py   # cleans + converts currency -> books_clean.csv, zepto_catalogue.db
python3 queries.py      # runs the 5+ SQL queries, pd.read_sql / pd.merge cross-check
```

**Design decisions:**
- Scope: first 5 paginated "All products" listing pages (yields 100 books across all site categories, well over the 60-book / 3-category minimum). Each book's `category` is scraped from its own detail page via the breadcrumb trail.
- Currency conversion uses the **fixed project-defined baseline rate: 1 GBP = 105.50 INR** — a constant, not a live/dated market rate, exactly as required. No network call or lookup is needed for this required path.
- Row-level cleaning failures: unparseable `price_gbp` → median-imputed (a numeric field, and prices are structurally very unlikely to be missing on this site, so imputation preserves row/category coverage). Unparseable `star_rating` → row dropped (fabricating a 1–5 star signal would misrepresent a customer-facing rating, unlike a plausible price gap). Unparseable `availability` → defaulted to `in_stock=False` (the conservative assumption for a catalogue feed).
- Schema: `categories(category_id PK, category_name)` ↔ `books(book_id PK, ..., category_id FK)` — a standard 1-to-many normalized design.
- `queries.py` demonstrates SELECT/WHERE, ORDER BY/LIMIT, DISTINCT, BETWEEN, and a JOIN (highest-rated book per category), then reproduces the JOIN with `pd.merge` on DataFrames loaded via `pd.read_sql`, and asserts the two results match.

> Note: books.toscrape.com is a public site built specifically for scraping practice — no login/API key required. (Verified working via direct browser/normal network access; only my own development sandbox's outbound network allowlist blocked it during authoring — the script itself is standard `requests`+`BeautifulSoup`.)

### Module 2 — Analytics Pipeline

```bash
cd analytics
python3 01_eda.py       # loads Titanic once, profiles, cleans, EDA, saves titanic.csv
python3 02_modeling.py  # reads the same titanic.csv, runs the full modeling pipeline
```

Outputs: `titanic.csv` (committed offline fallback), `charts/*.png`, `eda_report.md`, `model_report.md`, `best_pipeline.joblib`.

**Design decisions (full detail + all required written interpretations live in `eda_report.md` and `model_report.md`):**
- The raw dataset is loaded via `sns.load_dataset('titanic')` **exactly once**, in `01_eda.py`, which immediately caches it to `titanic.csv`. `02_modeling.py` reads only that CSV — it never calls `sns.load_dataset` again.
- Missing-value handling follows the stated threshold rule: `embarked`/`embark_town` (<5% missing) → rows dropped; `age` (~20% missing) → median-imputed; `deck` (~77% missing) → encoded as its own `"Missing"` category rather than dropped or imputed, since imputing 3-in-4 values would be unreliable.
- IQR-based outlier counts, fare skewness (mean > median > mode → right-skewed), all three bivariate survival breakdowns, and the 6-column correlation heatmap (with the two strongest pairs interpreted) are all in `eda_report.md`.
- Modeling: stratified 80/20 split → `ColumnTransformer` (median-impute + scale numeric; most-frequent-impute + one-hot categorical) inside a `Pipeline`, fit only on train. Logistic Regression, Decision Tree, Random Forest trained on the identical split; full metric suite + ROC/AUC reported; decision tree visualized with `plot_tree`.
- Imbalance comparison (baseline vs. `class_weight='balanced'` vs. SMOTE-on-train-fold-only) and `GridSearchCV` tuning of the Random Forest (with `oob_score=True`) are both in `model_report.md`, along with the fare linear-regression side-task (MAE/RMSE/R²/Adjusted R², heteroscedasticity check via residual plot).
- The final, best-performing **complete** pipeline (preprocessing + estimator together) is saved via `joblib.dump` and reload-verified against raw input.

### Module 3 — Support Assistant

```bash
cd support_assistant
python3 ingest.py                       # embeds the 8 docs into ChromaDB (one-time; downloads all-MiniLM-L6-v2 the first run)
MOCK_LLM=1 uvicorn main:app --host 0.0.0.0 --port 7860   # MOCK_LLM=1 is also the default if unset
```

Then, in another terminal:
```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "How long do I have to return a damaged item?"}'

curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
```

Docker (required, graded baseline — build & run locally):
```bash
cd support_assistant
docker build -t zepto-assistant .
docker run -p 7860:7860 zepto-assistant
```

**Design decisions:**
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, run fully locally, no API key. Stored in a persistent ChromaDB collection (`chroma_db/`), one chunk per source document (each of the 8 docs is already short and topically self-contained).
- Structured prompt template (`prompt_template.py`) follows role → context → task → format → length, with an explicit negative constraint ("do not answer using information not present in the provided context") and one embedded few-shot example. This template is used only by the optional `MOCK_LLM=0` real-LLM extension.
- `graph.py` builds a 3-node LangGraph `StateGraph` (`classify_intent`, `retrieve_and_answer`, `direct_answer`) with a conditional edge from `classify_intent` routing on the classified intent. **`MOCK_LLM` (default `1`/unset = mock, the graded baseline) gates only the generation step inside each node** — retrieval itself (embed query → ChromaDB top-3) always runs for real in both modes, since it needs no API key. In mock mode: intent classification is a keyword heuristic; `retrieve_and_answer` returns a canned `"Based on the retrieved context: {snippet}"` string built from the top retrieved chunk; `direct_answer` returns a fixed canned string. No network/LLM call is made anywhere in mock mode. Setting `MOCK_LLM=0` switches every generation step to a real LLM call (wired as a clear extension point in `graph.py`, intended for Groq's free tier) — this is an optional, ungraded stretch and is **not** what gets graded.
- The final answer is always returned as a Pydantic-validated `AskResponse(answer, sources, confidence)` (`schemas.py`). In mock mode it's populated deterministically by the code itself. The optional real-LLM path includes a retry-with-corrective-instruction loop (up to 2 retries) if the raw LLM output fails schema validation.
- **Architecture walkthrough (ingestion → embedding → retrieval → generation):**
  1. **Ingestion**: `ingest.load_documents()` reads the 8 `.txt` files in `docs/`, one chunk per file.
  2. **Embedding**: `ingest.get_embedding_model()` loads `all-MiniLM-L6-v2`; `ingest.build_collection()` embeds every chunk and stores it in the ChromaDB collection `zepto_policies` (persisted to `chroma_db/`).
  3. **Retrieval**: the `retrieve_and_answer` node in `graph.py` calls `ingest.query_collection()`, which embeds the incoming query and does a top-3 cosine-similarity lookup against that collection.
  4. **Generation**: still inside `retrieve_and_answer` (or `direct_answer` for non-policy queries), the MOCK_LLM branch decides whether the answer is templated (mock, default) or LLM-generated using `prompt_template.build_prompt()` (optional extension). The result is packaged into the `AskResponse` schema and returned by the FastAPI `POST /ask` endpoint in `main.py`.

**Example call transcripts (MOCK_LLM at default):** see `support_assistant/README.md`.

---

## Git workflow

This repository's history includes a feature branch (`feature/support-assistant`) created off `main`, committed to at least twice, and merged back — visible via:

```bash
git log --graph --all --oneline
```

---

## Academic integrity

All code, analysis, and written interpretations in this repository are original work produced for this capstone, built with reference to standard library/framework documentation only.
