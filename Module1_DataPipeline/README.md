# Module 1 — Data Pipeline (`/Module1_DataPipeline`)

Scrapes catalogue-style product data from [books.toscrape.com](https://books.toscrape.com)
(a public scraping-practice sandbox), cleans and types it, converts price to
INR at a fixed project baseline rate, and loads it into a normalized SQLite
database that is then queried with both SQL and pandas.

## Files

| File | Purpose |
|---|---|
| `scrape.py` | Scrapes 5 categories (Travel, Mystery, Historical Fiction, Classics, Romance) across all their pages, saves `books_raw.csv`. |
| `clean_and_load.py` | Cleans/types `books_raw.csv`, applies the fixed GBP→INR rate, writes `books_cleaned.csv`, and loads it into `zepto_catalogue.db`. |
| `queries.py` | Runs 7 SQL queries (3 of them JOINs) against the DB and cross-checks each JOIN against a pure-pandas `pd.merge`. Output saved in `queries_output.txt`. |
| `zepto_catalogue.db` | The generated SQLite database (regenerable by re-running the two scripts above). |

## How to run

```bash
pip install -r requirements.txt   # or the project's consolidated requirements.txt
python scrape.py            # needs internet access to books.toscrape.com
python clean_and_load.py    # cleans, converts currency, loads SQLite
python queries.py           # runs SQL + pandas verification, prints results
```

> **Sandbox note:** this development environment could not reach
> `books.toscrape.com` directly, so the wiring of `clean_and_load.py` and
> `queries.py` was validated end-to-end against a synthetic 70-row fixture
> in the exact shape `scrape.py` produces (same columns: `title`, `price`,
> `star_rating`, `availability`, `category`). Running `scrape.py` with a
> normal internet connection reproduces `books_raw.csv` for real and the
> rest of the pipeline runs unchanged on top of it.

## Design decisions

**Category / page scope.** Five full categories (Travel, Mystery, Historical
Fiction, Classics, Romance) are scraped across all their paginated listing
pages, comfortably clearing the 60-row minimum while satisfying the
"≥3 categories" requirement with margin.

**Cleaning / missing-value strategy.**
- `price_gbp` (numeric, feeds every downstream calculation): if a price
  string fails to parse, it is **median-imputed** rather than dropped —
  losing an entire row over one bad currency string would discard useful
  rating/category information for little benefit.
- `rating`, `in_stock`, `title`, `category` (categorical/text): a failed
  parse here is **dropped**, since imputing a star rating or stock status
  would fabricate information that doesn't exist in the source page.

**Currency conversion.** `price_inr = price_gbp * 105.50`. This is the
project's fixed, artificial baseline rate (not a live or historical market
rate), so it requires no API call, no key, and no date reference — it is
simply applied as a constant, as instructed.

**Schema.** Two tables, `categories(category_id PK, category_name)` and
`books(book_id PK, title, price_gbp, price_inr, rating, in_stock,
category_id FK)`, in 2NF: category names aren't repeated on every book row,
and every book links back to exactly one category via foreign key.

**SQL query coverage.** The 7 queries in `queries.py` cover `SELECT`/`WHERE`
(in-stock books), `ORDER BY` + `LIMIT` (top 10 priciest books), `DISTINCT`
(distinct ratings), `IN` + `BETWEEN` (rating in {4,5}, price 20–40 GBP), and
**three JOINs**: (5) books ranked by rating within each category, (6) average
price and book count per category (`JOIN` + `GROUP BY`), and (7) in-stock
mid-priced books with their category (`JOIN` + `WHERE` + `BETWEEN`). Each JOIN
is reproduced with `pd.read_sql` and independently with `pd.merge` on the two
DataFrames loaded from the tables, and the outputs are asserted equal.
