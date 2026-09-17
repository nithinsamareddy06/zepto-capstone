"""
Module 1 - Data Pipeline
Scrapes book data from books.toscrape.com across >=3 categories (>=60 rows),
cleans it, converts GBP->INR at a fixed baseline rate, and saves a raw CSV
that clean_and_load.py will consume.

books.toscrape.com is a public scraping-practice sandbox: no login, no API
key, no paid tier. Run this script with an active internet connection.
"""

import time
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://books.toscrape.com/"
CATALOGUE_URL = BASE_URL + "catalogue/"

# Categories chosen for this run (well-populated, gives us comfortably >60 rows)
CATEGORIES = [
    "travel_2",
    "mystery_3",
    "historical-fiction_4",
    "classics_6",
    "romance_8",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (capstone-project scraper; educational use)"}

RATING_WORD_TO_INT = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def get_soup(url):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def scrape_category(category_slug):
    """Scrape every book across every paginated page of one category."""
    rows = []
    page = 1
    while True:
        if page == 1:
            url = f"{CATALOGUE_URL}category/books/{category_slug}/index.html"
        else:
            url = f"{CATALOGUE_URL}category/books/{category_slug}/page-{page}.html"

        try:
            soup = get_soup(url)
        except requests.HTTPError:
            break  # no more pages for this category

        products = soup.find_all("article", class_="product_pod")
        if not products:
            break

        # human-readable category name from the sidebar breadcrumb
        breadcrumb = soup.find("ul", class_="breadcrumb")
        category_name = breadcrumb.find_all("li")[-1].get_text(strip=True) if breadcrumb else category_slug

        for product in products:
            title = product.h3.a["title"]
            price_text = product.find("p", class_="price_color").get_text(strip=True)
            rating_classes = product.find("p", class_="star-rating")["class"]
            rating_word = [c for c in rating_classes if c != "star-rating"][0]
            availability_text = product.find("p", class_="instock availability").get_text(strip=True)

            rows.append(
                {
                    "title": title,
                    "price": price_text,
                    "star_rating": rating_word,
                    "availability": availability_text,
                    "category": category_name,
                }
            )

        page += 1
        time.sleep(0.3)  # be polite to the sandbox server

    return rows


def main():
    all_rows = []
    for slug in CATEGORIES:
        print(f"Scraping category: {slug}")
        cat_rows = scrape_category(slug)
        print(f"  -> {len(cat_rows)} books found")
        all_rows.extend(cat_rows)

    df = pd.DataFrame(all_rows)
    print(f"\nTotal scraped rows: {len(df)} across {df['category'].nunique()} categories")
    assert len(df) >= 60, "Scraped fewer than 60 books - add another category."

    df.to_csv("books_raw.csv", index=False)
    print("Saved raw scrape to books_raw.csv")


if __name__ == "__main__":
    main()
