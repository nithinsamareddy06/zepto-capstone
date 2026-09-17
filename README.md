# Zepto Data & AI Platform — Capstone Project

One repository, three internally-linked capabilities, submitted as a single
project for the Certificate Program in AI/ML:

- **`/data_pipeline`** (25 marks) — scrape → clean → convert → store →
  query: a raw-to-relational data-engineering pipeline over public
  catalogue data.
- **`/analytics`** (50 marks) — a full analyst-to-data-scientist workflow
  on the Titanic dataset: profiling, cleaning, a visual data story, and a
  rigorously evaluated predictive-modeling pipeline (classification +
  regression).
- **`/support_assistant`** (25 marks) — a small, complete GenAI RAG
  service over Zepto's own policy documents: ChromaDB retrieval, a
  LangGraph-orchestrated flow, a Pydantic structured-output guarantee, and
  a FastAPI wrapper, gated behind a fully offline `MOCK_LLM` mode that is
  what gets graded.

Each module is graded independently against its own rubric, but they live
together in this one repo as a single, coherent submission.

## Repository layout

```
.
├── README.md                  <- this file
├── requirements.txt            <- consolidated deps for all 3 modules (see note below)
├── data_pipeline/
│   ├── scrape.py
│   ├── clean_and_load.py
│   ├── queries.py
│   ├── zepto_catalogue.db
│   ├── queries_output.txt
│   ├── books_raw.csv / books_cleaned.csv
│   └── README.md
├── analytics/
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── titanic.csv / titanic_cleaned.csv
│   ├── charts/
│   ├── best_pipeline.joblib
│   ├── EDA_FINDINGS.md / MODELING_REPORT.md
│   └── README.md
└── support_assistant/
    ├── docs/doc_01.txt … doc_08.txt
    ├── ingest.py / rag.py / prompt_template.py / graph.py / main.py
    ├── Dockerfile
    ├── requirements.txt
    └── README.md
```

## Setup

This project uses **one consolidated `requirements.txt`** at the repo
root, covering all 3 modules' dependencies (module-specific subsets are
also listed in each module's own README/requirements.txt if you'd rather
install per module).

```bash
python -m venv venv && source venv/bin/activate    # optional but recommended
pip install -r requirements.txt
```

## How to run each module end to end

**1. Data pipeline**
```bash
cd data_pipeline
python scrape.py            # needs internet access to books.toscrape.com
python clean_and_load.py    # cleans, converts currency, loads SQLite
python queries.py           # runs the 5 required SQL queries + pandas check
```

**2. Analytics**
```bash
cd analytics
python 01_eda.py        # loads Titanic once, profiles, cleans, EDA story
python 02_modeling.py   # classifiers, imbalance handling, tuning, regression, saved pipeline
```

**3. Support assistant**
```bash
cd support_assistant
python ingest.py                                  # builds the ChromaDB index
uvicorn main:app --host 0.0.0.0 --port 7860        # MOCK_LLM defaults to the graded offline mode
# or, via Docker:
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

## Design decisions — summary

**Module 1.** Five categories scraped across all their pages (comfortably
≥60 rows, ≥3 categories). Numeric parse failures (`price_gbp`) are
median-imputed; categorical/text parse failures (`rating`, `in_stock`,
`title`, `category`) are dropped, since fabricating a star rating or stock
status would misrepresent the source data. Currency conversion uses the
required fixed baseline (1 GBP = 105.50 INR) — an artificial project
constant, not a live rate. Schema is two tables (`categories`, `books`)
sharing a PK/FK relationship; 5 SQL queries cover every required clause
plus a JOIN, cross-checked against an equivalent `pd.merge`. Full
reasoning: [`data_pipeline/README.md`](data_pipeline/README.md).

**Module 2.** The Titanic dataset is loaded exactly once via
`sns.load_dataset`, cached to `titanic.csv`, cleaned per an explicit
missing-value threshold rule, and explored with univariate, bivariate,
correlation, and multivariate ("data story") analysis, each chart backed
by a written interpretation. The modeling pipeline continues from that
same cleaned data: a stratified split, leak-free `ColumnTransformer`
preprocessing (fit on train only), three classifiers with a full metric
suite, an imbalance-handling comparison (baseline vs. `class_weight`
vs. SMOTE), `GridSearchCV` + OOB-score tuning on Random Forest, a
`fare`-prediction regression side-task with a heteroscedasticity
check, and a final saved end-to-end `joblib` pipeline verified to
reload and predict correctly on raw input. Full reasoning and results:
[`analytics/README.md`](analytics/README.md), [`analytics/EDA_FINDINGS.md`](analytics/EDA_FINDINGS.md),
[`analytics/MODELING_REPORT.md`](analytics/MODELING_REPORT.md).

**Module 3.** 8 Zepto policy documents are chunked one-per-document,
embedded locally with `all-MiniLM-L6-v2`, and stored in ChromaDB. A
3-node LangGraph flow (`classify_intent` → `retrieve_and_answer` /
`direct_answer`) routes each query via a conditional edge; retrieval
always runs for real (no API key needed), while only the final
answer-generation step branches on the `MOCK_LLM` environment variable.
The graded baseline (`MOCK_LLM` unset/`1`) is fully deterministic and
offline — no LLM call anywhere — with the `answer`/`sources`/`confidence`
schema populated directly in code and validated by a Pydantic model. The
optional `MOCK_LLM=0` extension (a free-tier LLM such as Groq) is wired
up as a clearly-marked, ungraded stretch with schema-validation retries.
A FastAPI `POST /ask` endpoint and a Dockerfile expose the service
locally. Full architecture walkthrough and example call transcripts:
[`support_assistant/README.md`](support_assistant/README.md).

## Sandbox / environment notes

Two external resources used by this project were not reachable from the
development sandbox this repo was built in (`books.toscrape.com` for
Module 1, and `huggingface.co`'s model hub for Module 3's embedding
model). Both modules' code is written and structurally validated against
those exact interfaces (Module 1's cleaning/loading/query logic was
tested against a synthetic fixture in `scrape.py`'s output shape; Module
3's full ingestion/retrieval/graph/FastAPI flow was tested with a
temporary stand-in embedder). Module 2, whose only external dependency
(`raw.githubusercontent.com`, which serves Seaborn's dataset repository)
*was* reachable, was run for real end-to-end, and all of its charts,
tables, and reports in this repo reflect a genuine run against the actual
Titanic dataset. See each module's own README for the specific note and
what to expect when you run it with normal internet access.

## Git workflow

This repository's history includes a feature branch
(`feature/analytics-module`, containing Module 2 and Module 3 work across
2 commits) merged back into `master` via an explicit merge commit — visible
with `git log --oneline --graph --all`.
