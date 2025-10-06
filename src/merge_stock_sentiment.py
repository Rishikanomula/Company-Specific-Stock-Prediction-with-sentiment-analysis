import pandas as pd

# Load data
news_df = pd.read_csv(r"C:\Rishika\SPP\data\live_news_with_finbert.csv")
news_df['date'] = pd.to_datetime(news_df['date'])

stock_df = pd.read_csv(r"C:\Rishika\SPP\data\HDFCBANK_data.csv")
stock_df['Date'] = pd.to_datetime(stock_df['Date'])

# Filter Reliance news
company = "Reliance Industries"
company_news = news_df[news_df['company'] == company]

# Average sentiment per date
daily_sentiment = company_news.groupby('date')['finbert_sentiment'].mean().reset_index()

# Merge
merged_df = pd.merge(stock_df, daily_sentiment, how='left', left_on='Date', right_on='date')
merged_df['finbert_sentiment'] = merged_df['finbert_sentiment'].fillna(0)
merged_df.drop(columns=['date'], inplace=True)

# Save merged CSV
merged_df.to_csv("../data/HDFCBANK_merged.csv", index=False)
print("✅ Saved merged data for Reliance Industries → RELIANCE_merged.csv")
