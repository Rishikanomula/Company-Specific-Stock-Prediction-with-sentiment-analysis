import requests
import pandas as pd
import time
from transformers import pipeline

# Your NewsData.io API key
API_KEY = "pub_31797144e10748958eeef09791d43e70"

# Define the companies and date range
companies = ["TCS", "Reliance", "HDFCBank", "Infosys"]
from_date = "2020-01-01"
to_date = "2025-10-03"

# Base URL for NewsData.io API
base_url = "https://newsdata.io/api/1/news"

# Initialize FinBERT sentiment analyzer
sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# Function to fetch news articles for a given query
def fetch_news(company, from_date, to_date, api_key):
    all_articles = []
    page = 1
    while True:
        params = {
            'apikey': api_key,
            'q': company,
            'from_date': from_date,
            'to_date': to_date,
            'language': 'en',
            'page': page
        }
        response = requests.get(base_url, params=params)
        data = response.json()

        if data.get("status") != "success" or "results" not in data:
            print(f"No data or error for {company} on page {page}")
            break

        articles = data["results"]
        if not articles:
            break

        all_articles.extend(articles)

        if page >= data.get("totalPages", page):
            break

        page += 1
        time.sleep(1)  # To avoid API rate limits

    return all_articles

# Function to analyze sentiment scores
def analyze_sentiments(titles):
    sentiments = []
    scores = []
    batch_size = 16  # process in batches for efficiency
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i+batch_size]
        results = sentiment_pipeline(batch)
        for res in results:
            sentiments.append(res['label'])
            scores.append(res['score'])
    return sentiments, scores

# Collect news, analyze sentiment, and save separately
for company in companies:
    print(f"Fetching news for {company}...")
    articles = fetch_news(company, from_date, to_date, API_KEY)
    if articles:
        df_company = pd.DataFrame(articles)
        if 'title' in df_company.columns:
            # Fill missing titles with empty string to avoid errors
            df_company['title'] = df_company['title'].fillna('')

            # Analyze sentiment of titles
            sentiments, scores = analyze_sentiments(df_company['title'].tolist())

            df_company['sentiment'] = sentiments
            df_company['sentiment_score'] = scores

        else:
            df_company['sentiment'] = None
            df_company['sentiment_score'] = None

        # Save each company's data separately
        df_company.to_csv(f"C:\Rishika\SPP\data\{company}_news_sentiments_2020_2025.csv", index=False)
        print(f"Saved {company} news with sentiment.")
    else:
        print(f"No articles found for {company}.")

print("All company news data with sentiments saved.")
