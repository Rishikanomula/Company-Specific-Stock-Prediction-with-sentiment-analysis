import pandas as pd

# ========================
# File paths
# ========================
stock_file = r"C:\Rishika\SPP\data\HDFCBANK_data.csv"
sentiment_file = r"C:\Rishika\SPP\hdfc bank news\hdfc_sentiment_news.csv"
output_file = r"C:\Rishika\SPP\hdfc bank news\merged_hdfc_stock_sentiment.csv"

# ========================
# Load both datasets
# ========================
stock_df = pd.read_csv(stock_file)
sent_df = pd.read_csv(sentiment_file)

# ========================
# Clean and convert date columns
# ========================

# Stock file: 'Date' already in YYYY-MM-DD format
stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')

# Sentiment file: convert datetime and extract only the date portion
sent_df['date'] = pd.to_datetime(sent_df['date'], errors='coerce')
sent_df['date_only'] = sent_df['date'].dt.date

# ========================
# Aggregate sentiment per date (in case of multiple headlines)
# ========================
# Take average compound score and most frequent sentiment label per day
sentiment_daily = (
    sent_df.groupby('date_only')
    .agg({
        'compound_score': 'mean',
        'sentiment': lambda x: x.value_counts().index[0]  # majority sentiment
    })
    .reset_index()
    .rename(columns={'date_only': 'Date'})
)

# Convert 'Date' to datetime for merging
sentiment_daily['Date'] = pd.to_datetime(sentiment_daily['Date'], errors='coerce')

# ========================
# Merge stock data with daily sentiment
# ========================
merged_df = pd.merge(stock_df, sentiment_daily, on='Date', how='left')

# ========================
# Fill missing sentiment values with Neutral and 0.0
# ========================
merged_df['sentiment'] = merged_df['sentiment'].fillna('Neutral')
merged_df['compound_score'] = merged_df['compound_score'].fillna(0.0)

# ========================
# Save merged dataset
# ========================
merged_df.to_csv(output_file, index=False)
print(f"✅ Merging complete. Saved to: {output_file}")

# ========================
# Preview result
# ========================
print("\n📊 Merged Data Preview:")
print(merged_df.head(10))
