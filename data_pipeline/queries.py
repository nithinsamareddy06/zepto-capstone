"""
Module 1 - Data Pipeline
Runs the required SQL queries against zepto_catalogue.db, prints their
output, then reproduces two of them in pandas (pd.read_sql + pd.merge)
to show both approaches agree.

Run: python3 queries.py
"""

import sqlite3
import pandas as pd

DB_PATH = "zepto_catalogue.db"


def run(cur, label, sql):
    print(f"\n--- {label} ---")
    print(sql.strip())
    rows = cur.execute(sql).fetchall()
    for r in rows:
        print(r)
    return rows


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. SELECT / WHERE
    run(cur, "Q1: SELECT/WHERE - books in stock under 20 GBP",
        "SELECT title, price_gbp FROM books WHERE in_stock = 1 AND price_gbp < 20;")

    # 2. ORDER BY + LIMIT
    run(cur, "Q2: ORDER BY/LIMIT - 10 most expensive books (INR)",
        "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10;")

    # 3. DISTINCT
    run(cur, "Q3: DISTINCT - all category names in use",
        "SELECT DISTINCT category_name FROM categories;")

    # 4. IN / BETWEEN
    run(cur, "Q4: BETWEEN - books priced between 10 and 30 GBP",
        "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 30;")

    # 5. JOIN - top-rated books per category (rank 1 per category here shown
    #    as the single highest-rated book per category, ties broken by title)
    join_sql = """
        SELECT c.category_name, b.title, b.rating
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.rating = (
            SELECT MAX(b2.rating) FROM books b2 WHERE b2.category_id = b.category_id
        )
        ORDER BY c.category_name, b.title
        LIMIT 15;
    """
    run(cur, "Q5: JOIN - highest-rated book(s) per category", join_sql)

    # --- pd.read_sql for two of the above queries ---
    print("\n--- pandas: pd.read_sql for Q1 and Q5 ---")
    df_q1_sql = pd.read_sql("SELECT title, price_gbp FROM books WHERE in_stock=1 AND price_gbp<20;", conn)
    df_join_sql = pd.read_sql(join_sql, conn)
    print(df_q1_sql.head())
    print(df_join_sql.head())

    # --- Reproduce the join query with pd.merge on in-memory DataFrames ---
    print("\n--- pandas: reproducing the JOIN with pd.merge (no SQL) ---")
    books_df = pd.read_sql("SELECT * FROM books;", conn)
    categories_df = pd.read_sql("SELECT * FROM categories;", conn)

    merged = books_df.merge(categories_df, on="category_id", how="inner")
    max_rating_per_cat = merged.groupby("category_id")["rating"].transform("max")
    df_join_pandas = (
        merged[merged["rating"] == max_rating_per_cat]
        [["category_name", "title", "rating"]]
        .sort_values(["category_name", "title"])
        .head(15)
        .reset_index(drop=True)
    )
    print(df_join_pandas)

    # --- Equivalence check ---
    sql_result = df_join_sql.reset_index(drop=True)
    pandas_result = df_join_pandas.reset_index(drop=True)
    equivalent = sql_result.equals(pandas_result)
    print(f"\nSQL join result matches pandas merge result: {equivalent}")

    conn.close()


if __name__ == "__main__":
    main()
