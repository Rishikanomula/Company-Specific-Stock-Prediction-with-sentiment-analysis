# nyt_1_article_per_day_scraper.py
from pynytimes import NYTAPI
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import logging

# -----------------------------
# CONFIGURATION
# -----------------------------

# Load NYT API key (set via environment variable or paste directly)
API_KEY = os.getenv("NYT_API_KEY", "ydGO4lKT8U2VxtKp8TaqgMuvLZ50QShq")

# Initialize NYT API client
nyt = NYTAPI(API_KEY, parse_dates=True)

# Configure logging
logging.basicConfig(
    filename="nyt_daily_scraper.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Search terms and sections
SEARCH_TERMS = "apple iphone ipad macbook"
SECTIONS = ["Technology", "Business Day", "Opinion"]

# Date range: 22-Oct-2024 → 22-Oct-2025
START_DATE = datetime(2024, 10, 22)
END_DATE = datetime(2025, 10, 22)

# -----------------------------
# HELPER FUNCTION
# -----------------------------

def fetch_one_article_for_day(date):
    """Fetch the most relevant article for a specific day."""
    try:
        fq_query = " OR ".join([f'section_name:("{section}")' for section in SECTIONS])
        
        data = nyt.article_search(
            query=SEARCH_TERMS,
            results=1,  # only one per day
            dates={"begin": date, "end": date},
            options={
                "sort": "relevance",  # could use "newest" instead
                "fq": fq_query
            }
        )

        if not data:
            logging.info(f"No article found for {date.date()}")
            return None

        item = data[0]
        return {
            "headline": item.get("headline", {}).get("main"),
            "snippet": item.get("abstract"),
            "source": item.get("source"),
            "pub_date": item.get("pub_date"),
            "web_url": item.get("web_url"),
            "keywords": [k["value"] for k in item.get("keywords", [])],
            "section_name": item.get("section_name"),
            "search_date": date.date()
        }

    except Exception as e:
        logging.error(f"Error fetching {date.date()}: {e}")
        time.sleep(5)
        return None

# -----------------------------
# MAIN FUNCTION
# -----------------------------

def collect_daily_articles(start, end):
    """Fetch 1 article per day between start and end dates."""
    current = start
    all_articles = []

    while current <= end:
        logging.info(f"Fetching article for {current.date()}...")
        article = fetch_one_article_for_day(current)
        if article:
            all_articles.append(article)
        time.sleep(2)  # respect NYT API rate limits
        current += timedelta(days=1)

    return all_articles

# -----------------------------
# MAIN SCRIPT
# -----------------------------

if __name__ == "__main__":
    logging.info(f"NYT daily scraper started for {START_DATE.date()} → {END_DATE.date()}")

    articles = collect_daily_articles(START_DATE, END_DATE)
    df = pd.DataFrame(articles)

    output_file = "nyt_daily_articles_2024_10_22_to_2025_10_22.csv"
    df.to_csv(output_file, index=False)

    logging.info(f"✅ Saved {len(df)} daily articles to {output_file}")
    print(f"✅ Done! Saved {len(df)} daily articles to {output_file}")
