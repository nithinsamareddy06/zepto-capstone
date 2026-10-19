"""
Module 1 - Data Pipeline
Runs >=5 SQL queries against zepto_catalogue.db covering SELECT/WHERE,
ORDER BY, LIMIT, DISTINCT, IN/BETWEEN, and a JOIN. Then reproduces two of
them via pandas (pd.read_sql and pd.merge) and shows the outputs match.

Run: python queries.py   (after clean_and_load.py has created the DB)
"""

import sqlite3
import pandas as pd

DB_PATH = "zepto_catalogue.db"


def run_query(conn, label, sql):
    print(f"\n{'=' * 70}\nQuery: {label}\n{'-' * 70}\n{sql.strip()}\n{'-' * 70}")
    df = pd.read_sql(sql, conn)
    print(df.to_string(index=False))
    return df


def main():
    conn = sqlite3.connect(DB_PATH)

    # 1) SELECT / WHERE
    q1 = "SELECT title, price_gbp, rating FROM books WHERE in_stock = 1;"
    run_query(conn, "1. In-stock books (SELECT + WHERE)", q1)

    # 2) ORDER BY + LIMIT
    q2 = "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10;"
    run_query(conn, "2. Top 10 most expensive books, in INR (ORDER BY + LIMIT)", q2)

    # 3) DISTINCT
    q3 = "SELECT DISTINCT rating FROM books ORDER BY rating;"
    run_query(conn, "3. Distinct rating values present (DISTINCT)", q3)

    # 4) IN / BETWEEN
    q4 = """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating IN (4, 5) AND price_gbp BETWEEN 20 AND 40;
    """
    run_query(conn, "4. Highly-rated, mid-priced books (IN + BETWEEN)", q4)

    # 5) JOIN - top 3 highest-rated books per category
    q5 = """
        SELECT c.category_name, b.title, b.rating, b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY c.category_name, b.rating DESC, b.price_inr DESC;
    """
    join_df = run_query(conn, "5. Book/category JOIN, ranked by rating within category", q5)


    # 6) JOIN - average price and book count per category
    q6 = """
        SELECT c.category_name, COUNT(*) AS n_books,
               ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        GROUP BY c.category_name
        ORDER BY avg_price_gbp DESC;
    """
    run_query(conn, "6. Average price per category (JOIN + GROUP BY)", q6)

    # 7) JOIN - in-stock books priced between 20 and 40 GBP with category name
    q7 = """
        SELECT c.category_name, b.title, b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.in_stock = 1 AND b.price_gbp BETWEEN 20 AND 40
        ORDER BY b.price_gbp DESC
        LIMIT 15;
    """
    run_query(conn, "7. In-stock mid-priced books with category (JOIN + WHERE + BETWEEN)", q7)

    # --- pd.read_sql vs pd.merge equivalence check --------------------------
    print(f"\n{'=' * 70}\nEquivalence check: pd.read_sql JOIN vs pandas-only pd.merge\n{'=' * 70}")

    read_sql_join = pd.read_sql(
        """
        SELECT c.category_name, b.title, b.rating, b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        """,
        conn,
    ).sort_values(["category_name", "title"]).reset_index(drop=True)

    books_df = pd.read_sql("SELECT * FROM books", conn)
    categories_df = pd.read_sql("SELECT * FROM categories", conn)

    merge_join = (
        pd.merge(books_df, categories_df, on="category_id", how="inner")
        [["category_name", "title", "rating", "price_inr"]]
        .sort_values(["category_name", "title"])
        .reset_index(drop=True)
    )

    are_equal = read_sql_join.equals(merge_join)
    print(f"\npd.read_sql result (first 5 rows):\n{read_sql_join.head().to_string(index=False)}")
    print(f"\npd.merge result (first 5 rows):\n{merge_join.head().to_string(index=False)}")
    print(f"\nDataFrames are equal: {are_equal}")
    assert are_equal, "pd.read_sql and pd.merge outputs diverged!"


    # --- Equivalence for JOIN queries 6 and 7 -------------------------------
    agg_sql = pd.read_sql(q6, conn).sort_values("category_name").reset_index(drop=True)
    agg_merge = (
        pd.merge(books_df, categories_df, on="category_id")
        .groupby("category_name")
        .agg(n_books=("title", "count"), avg_price_gbp=("price_gbp", "mean"))
        .round({"avg_price_gbp": 2})
        .reset_index()
        .sort_values("category_name")
        .reset_index(drop=True)
    )
    ok6 = agg_sql.equals(agg_merge.astype(agg_sql.dtypes.to_dict()))
    print(f"\nQuery 6 pd.read_sql vs pd.merge+groupby equal: {ok6}")
    assert ok6

    q7_sql = pd.read_sql(q7, conn)
    q7_merge = (
        pd.merge(books_df, categories_df, on="category_id")
        .query("in_stock == 1 and 20 <= price_gbp <= 40")
        .sort_values("price_gbp", ascending=False)
        [["category_name", "title", "price_gbp"]]
    )
    key = ["price_gbp", "category_name", "title"]
    q7_merge = q7_merge.sort_values(key, ascending=[False, True, True]).head(15)
    ok7 = (q7_sql.sort_values(key, ascending=[False, True, True]).reset_index(drop=True)
           .equals(q7_merge.reset_index(drop=True)))
    print(f"Query 7 pd.read_sql vs pd.merge+filter equal: {ok7}")
    assert ok7

    conn.close()


if __name__ == "__main__":
    main()
