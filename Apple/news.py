# nyt_2020_to_today_scraper.py
from pynytimes import NYTAPI
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import logging

# -----------------------------
# CONFIGURATION
# -----------------------------

# Load NYT API key (set via: export NYT_API_KEY="your_key_here")
API_KEY = os.getenv("NYT_API_KEY", "ydGO4lKT8U2VxtKp8TaqgMuvLZ50QShq")

# Initialize NYTimes API client
nyt = NYTAPI(API_KEY, parse_dates=True)

# Configure logging
logging.basicConfig(
    filename="nyt_scraper.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Search terms and filters
SEARCH_TERMS = "apple iphone ipad macbook"
SECTIONS = ["Technology", "Business Day", "Opinion"]

# Date range: 2020-01-01 to today
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2025,10,21)

# -----------------------------
# HELPER FUNCTION TO FETCH ARTICLES
# -----------------------------

def fetch_month_articles(year, month):
    """Fetch all articles for a given month using NYT API."""
    try:
        begin = datetime(year, month, 1)
        # Next month (end exclusive)
        if month == 12:
            end = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end = datetime(year, month + 1, 1) - timedelta(days=1)

        logging.info(f"Fetching {year}-{month:02d} ...")

        data = nyt.article_search(
            query=SEARCH_TERMS,
            results=100,
            sections=SECTIONS,
            dates={"begin": begin, "end": end},
            options={"sort": "newest"}
        )

        articles = []
        for item in data:
            articles.append({
                "headline": item.get("headline", {}).get("main"),
                "snippet": item.get("abstract"),
                "source": item.get("source"),
                "pub_date": item.get("pub_date"),
                "web_url": item.get("web_url"),
                "keywords": [k["value"] for k in item.get("keywords", [])],
                "section_name": item.get("section_name")
            })

        logging.info(f"Collected {len(articles)} articles from {year}-{month:02d}")
        time.sleep(2)  # obey NYT rate limits
        return articles

    except Exception as e:
        logging.error(f"Error on {year}-{month:02d}: {e}")
        time.sleep(5)
        return []

# -----------------------------
# BATCH FETCH LOOP
# -----------------------------

def collect_all_articles(start, end):
    current = datetime(start.year, start.month, 1)
    all_articles = []

    while current <= end:
        month_articles = fetch_month_articles(current.year, current.month)
        all_articles.extend(month_articles)

        # move to next month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    return all_articles

# -----------------------------
# MAIN LOGIC
# -----------------------------

if __name__ == "__main__":
    logging.info("NYT Scraper started for 2020-01-01 to present...")

    articles = collect_all_articles(START_DATE, END_DATE)
    df = pd.DataFrame(articles)
    df.to_csv("nyt_apple_articles_2020_to_today.csv", index=False)

    logging.info(f"Saved {len(df)} articles total.")
    print(f"✅ Saved {len(df)} articles from {START_DATE.date()} to {END_DATE.date()}")
