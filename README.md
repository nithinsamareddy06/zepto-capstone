# Zepto Data & AI Platform — Capstone Project

One repository, three internally-linked capabilities, submitted as a single
project for the Certificate Program in AI/ML:

- **`/Module1_DataPipeline`** (25 marks) — scrape → clean → convert → store →
  query: a raw-to-relational data-engineering pipeline over public
  catalogue data.
- **`/Module2_ML_Analytics`** (50 marks) — a full analyst-to-data-scientist workflow
  on the Titanic dataset: profiling, cleaning, a visual data story, and a
  rigorously evaluated predictive-modeling pipeline (classification +
  regression).
- **`/Module3_Support_Assistant`** (25 marks) — a small, complete GenAI RAG
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
├── Module1_DataPipeline/
│   ├── scrape.py
│   ├── clean_and_load.py
│   ├── queries.py
│   ├── zepto_catalogue.db
│   ├── queries_output.txt
│   ├── books_raw.csv / books_cleaned.csv
│   └── README.md
├── Module2_ML_Analytics/
│   ├── 01_eda.py
│   ├── 02_modeling.py
│   ├── titanic.csv / titanic_cleaned.csv
│   ├── charts/
│   ├── best_pipeline.joblib
│   ├── EDA_FINDINGS.md / MODELING_REPORT.md
│   └── README.md
└── Module3_Support_Assistant/
    ├── docs/doc_01.txt … doc_08.txt
    ├── ingest.py / rag.py / prompt_template.py / graph.py / main.py
    ├── Dockerfile
    ├── requirements.txt
    └── README.md
```

## Setup

This project uses a **`requirements.txt` in each module folder**, plus a
consolidated one at the repo root covering everything.

```bash
python -m venv venv && source venv/bin/activate    # optional but recommended
pip install -r requirements.txt
```

## How to run each module end to end

**1. Data pipeline**
```bash
cd Module1_DataPipeline
python scrape.py            # needs internet access to books.toscrape.com
python clean_and_load.py    # cleans, converts currency, loads SQLite
python queries.py           # runs 7 SQL queries (3 JOINs) + pandas checks
```

**2. Analytics**
```bash
cd Module2_ML_Analytics
python 01_eda.py        # loads Titanic once, profiles, cleans, EDA story
python 02_modeling.py   # classifiers, imbalance handling, tuning, regression, saved pipeline
```

**3. Support assistant**
```bash
cd Module3_Support_Assistant
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
sharing a PK/FK relationship; 7 SQL queries cover every required clause
plus 3 JOINs, each cross-checked against an equivalent `pd.merge`. Full
reasoning: [`Module1_DataPipeline/README.md`](Module1_DataPipeline/README.md).

**Module 2.** The Titanic dataset is loaded exactly once via
`sns.load_dataset`, cached to `titanic.csv`, cleaned per an explicit
missing-value threshold rule (columns over 30% missing, like `deck`, are dropped), and explored with univariate, bivariate,
correlation, and multivariate ("data story") analysis, each chart backed
by a written interpretation. The modeling pipeline continues from that
same cleaned data: a stratified split, leak-free `ColumnTransformer`
preprocessing (fit on train only), three classifiers with a full metric
suite, an imbalance-handling comparison (baseline vs. `class_weight`
vs. SMOTE), `GridSearchCV` + OOB-score tuning on Random Forest, a
`fare`-prediction regression side-task with a heteroscedasticity
check, and a final saved end-to-end `joblib` pipeline verified to
reload and predict correctly on raw input. Full reasoning and results:
[`Module2_ML_Analytics/README.md`](Module2_ML_Analytics/README.md), [`Module2_ML_Analytics/EDA_FINDINGS.md`](Module2_ML_Analytics/EDA_FINDINGS.md),
[`Module2_ML_Analytics/MODELING_REPORT.md`](Module2_ML_Analytics/MODELING_REPORT.md).

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
[`Module3_Support_Assistant/README.md`](Module3_Support_Assistant/README.md).

All three modules were run end to end; outputs in this repo come from real runs.

## Git workflow

This repository's history includes a feature branch
(`feature/analytics-module`, containing Module 2 and Module 3 work across
2 commits) merged back into `master` via an explicit merge commit — visible
with `git log --oneline --graph --all`.
