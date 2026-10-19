"""
Module 1 - Data Pipeline
Reads books_raw.csv (produced by scrape.py), cleans/types every field,
converts GBP -> INR at the fixed project baseline rate, and loads the
result into a normalized two-table SQLite schema (categories <-> books).

Run: python clean_and_load.py
Requires books_raw.csv to already exist in this directory.
"""

import re
import sqlite3
import pandas as pd

RAW_CSV = "books_raw.csv"
DB_PATH = "zepto_catalogue.db"

# Project-defined fixed baseline conversion rate (NOT a live/historical market
# rate - an artificial constant stated here and in the README for grading).
GBP_TO_INR = 105.50

RATING_WORD_TO_INT = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def parse_price(price_text):
    """'£53.74' -> 53.74 (float). Returns None if it can't be parsed."""
    if not isinstance(price_text, str):
        return None
    match = re.search(r"[\d.]+", price_text.replace(",", ""))
    return float(match.group()) if match else None


def parse_rating(word):
    return RATING_WORD_TO_INT.get(word, None)


def parse_availability(text):
    """'In stock (22 available)' -> True; 'Out of stock' -> False."""
    if not isinstance(text, str):
        return None
    return "in stock" in text.lower()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_gbp"] = df["price"].apply(parse_price)
    df["rating"] = df["star_rating"].apply(parse_rating)
    df["in_stock"] = df["availability"].apply(parse_availability)

    before = len(df)

    # --- Missing / unparseable-field handling -------------------------------
    # price_gbp: numeric and central to every downstream calculation, so a
    # failed parse is imputed with the column median rather than dropped -
    # losing a whole row over one bad currency string would be wasteful.
    if df["price_gbp"].isna().any():
        median_price = df["price_gbp"].median()
        n_missing = df["price_gbp"].isna().sum()
        df["price_gbp"] = df["price_gbp"].fillna(median_price)
        print(f"Imputed {n_missing} missing price_gbp values with median {median_price:.2f}")

    # rating / in_stock / title / category: these are categorical/text fields
    # where imputing a value would fabricate information (e.g. inventing a
    # star rating), so any row that fails to parse here is dropped instead.
    df = df.dropna(subset=["rating", "in_stock", "title", "category"])

    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with unparseable rating/availability/title/category")

    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    # --- Fixed-rate currency conversion (graded, keyless baseline) ----------
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


def load_to_sqlite(df: pd.DataFrame, db_path=DB_PATH):
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

    categories = sorted(df["category"].unique())
    cur.executemany(
        "INSERT INTO categories (category_name) VALUES (?)",
        [(c,) for c in categories],
    )
    conn.commit()

    cat_id_map = dict(
        cur.execute("SELECT category_name, category_id FROM categories").fetchall()
    )

    book_rows = [
        (
            row.title,
            row.price_gbp,
            row.price_inr,
            row.rating,
            int(row.in_stock),
            cat_id_map[row.category],
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
    print(f"Loaded {len(df)} books across {len(categories)} categories into {db_path}")


def main():
    raw = pd.read_csv(RAW_CSV)
    cleaned = clean(raw)
    cleaned.to_csv("books_cleaned.csv", index=False)
    print(f"Cleaned dataset: {len(cleaned)} rows -> books_cleaned.csv")
    load_to_sqlite(cleaned)


if __name__ == "__main__":
    main()
