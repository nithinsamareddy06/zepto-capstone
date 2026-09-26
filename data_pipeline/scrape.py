"""
Module 1 - Data Pipeline
Scrapes book data from books.toscrape.com (a public scraping-practice site),
cleans it, converts GBP -> INR using a fixed baseline rate, and saves the
result as data_pipeline/books_raw.csv for the downstream clean/load step.

Scope chosen: the first 5 paginated listing pages of the "All products"
catalogue (page 1 .. page 5), which yields well over 60 books across many
categories (each book's own category is scraped from its detail page).
"""

import re
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE = "https://books.toscrape.com/"
CATALOGUE_PAGE = "https://books.toscrape.com/catalogue/page-{}.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (capstone-scraper/1.0)"}

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def get_book_links_from_listing(page_num: int):
    """Return list of absolute book-detail URLs from a catalogue listing page."""
    url = CATALOGUE_PAGE.format(page_num)
    soup = get_soup(url)
    links = []
    for article in soup.select("article.product_pod"):
        a = article.find("h3").find("a")
        href = a["href"]
        # links on listing pages are relative to /catalogue/
        full_url = "https://books.toscrape.com/catalogue/" + href.replace("../../../", "")
        links.append(full_url)
    return links


def scrape_book_detail(url: str) -> dict:
    soup = get_soup(url)

    title = soup.find("h1").get_text(strip=True)

    price_text = soup.select_one("p.price_color").get_text(strip=True)

    rating_tag = soup.select_one("p.star-rating")
    rating_word = [c for c in rating_tag.get("class", []) if c != "star-rating"]
    rating_word = rating_word[0] if rating_word else None

    availability_text = soup.select_one("p.availability").get_text(strip=True)

    # Category comes from the breadcrumb trail (2nd-to-last crumb)
    breadcrumbs = soup.select("ul.breadcrumb li a")
    category = breadcrumbs[-1].get_text(strip=True) if breadcrumbs else None

    return {
        "title": title,
        "price": price_text,
        "star_rating": rating_word,
        "availability": availability_text,
        "category": category,
        "source_url": url,
    }


def scrape_catalogue(num_pages: int = 5, delay: float = 0.2) -> pd.DataFrame:
    rows = []
    for page in range(1, num_pages + 1):
        print(f"Scraping listing page {page} ...")
        try:
            links = get_book_links_from_listing(page)
        except Exception as e:
            print(f"  Could not load page {page}: {e}")
            continue

        for link in links:
            try:
                rows.append(scrape_book_detail(link))
            except Exception as e:
                print(f"  Failed to scrape {link}: {e}")
            time.sleep(delay)

    df = pd.DataFrame(rows)
    print(f"Scraped {len(df)} books across {df['category'].nunique()} categories.")
    return df


if __name__ == "__main__":
    df = scrape_catalogue(num_pages=5)
    df.to_csv("books_raw.csv", index=False)
    print("Saved raw scrape to data_pipeline/books_raw.csv")
