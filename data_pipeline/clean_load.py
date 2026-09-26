"""
Module 1 - Data Pipeline
Reads data_pipeline/books_raw.csv (produced by scrape.py), cleans and types
every field, converts GBP -> INR at the fixed project rate, and loads the
result into a normalized SQLite database (data_pipeline/zepto_catalogue.db)
with a categories <-> books primary/foreign-key relationship.

Run: python3 clean_load.py
"""

import re
import sqlite3
import pandas as pd

RAW_CSV = "books_raw.csv"
DB_PATH = "zepto_catalogue.db"

# Fixed, project-defined baseline conversion rate (NOT a live market rate).
GBP_TO_INR = 105.50

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def clean_price(raw_price: str):
    """'£51.77' -> 51.77 (float). Returns None if it can't be parsed."""
    if not isinstance(raw_price, str):
        return None
    match = re.search(r"[\d.]+", raw_price)
    return float(match.group()) if match else None


def clean_rating(raw_rating: str):
    """'Three' -> 3 (int). Returns None if not a recognised rating word."""
    return RATING_WORDS.get(str(raw_rating).strip(), None)


def clean_availability(raw_availability: str):
    """'In stock (22 available)' -> True; 'Out of stock' -> False."""
    if not isinstance(raw_availability, str):
        return None
    return "in stock" in raw_availability.lower()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_gbp"] = df["price"].apply(clean_price)
    df["rating"] = df["star_rating"].apply(clean_rating)
    df["in_stock"] = df["availability"].apply(clean_availability)

    # --- Handle rows that failed to parse ---
    # price_gbp: numeric field -> median imputation (keeps the row / category
    #   mix intact; a scraped price is very unlikely to be missing, but if a
    #   listing page changes format mid-scrape this keeps the pipeline alive).
    if df["price_gbp"].isna().any():
        median_price = df["price_gbp"].median()
        n_missing = df["price_gbp"].isna().sum()
        print(f"Imputing {n_missing} missing price_gbp value(s) with median {median_price:.2f}")
        df["price_gbp"] = df["price_gbp"].fillna(median_price)

    # rating: a missing/unrecognised star-rating word is dropped rather than
    #   imputed, because "guessing" a 1-5 star rating would fabricate a
    #   customer-facing signal rather than a plausible price/availability gap.
    if df["rating"].isna().any():
        n_missing = df["rating"].isna().sum()
        print(f"Dropping {n_missing} row(s) with unparseable star rating")
        df = df.dropna(subset=["rating"])

    # in_stock: default unparseable availability text to False (treat unknown
    #   stock as "cannot confirm available" - the conservative choice for a
    #   catalogue feed).
    if df["in_stock"].isna().any():
        n_missing = df["in_stock"].isna().sum()
        print(f"Defaulting {n_missing} row(s) with unparseable availability to in_stock=False")
        df["in_stock"] = df["in_stock"].fillna(False)

    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    # --- Fixed-rate currency conversion (required, graded baseline) ---
    # 1 GBP = 105.50 INR -- a fixed, project-defined constant, not a live rate.
    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


def build_database(df: pd.DataFrame, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript(
        """
        DROP TABLE IF EXISTS books;
        DROP TABLE IF EXISTS categories;

        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        );
        """
    )

    categories = sorted(df["category"].dropna().unique())
    cur.executemany(
        "INSERT INTO categories (category_name) VALUES (?)",
        [(c,) for c in categories],
    )
    conn.commit()

    cat_map = dict(
        cur.execute("SELECT category_name, category_id FROM categories").fetchall()
    )

    book_rows = [
        (
            row.title,
            row.price_gbp,
            row.price_inr,
            int(row.rating),
            int(row.in_stock),
            cat_map[row.category],
        )
        for row in df.itertuples(index=False)
    ]
    cur.executemany(
        """INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        book_rows,
    )
    conn.commit()
    conn.close()
    print(f"Loaded {len(book_rows)} books across {len(categories)} categories into {db_path}")


if __name__ == "__main__":
    raw = pd.read_csv(RAW_CSV)
    cleaned = clean_dataframe(raw)
    cleaned.to_csv("books_clean.csv", index=False)
    print(f"Saved cleaned data to books_clean.csv ({len(cleaned)} rows)")
    build_database(cleaned)
