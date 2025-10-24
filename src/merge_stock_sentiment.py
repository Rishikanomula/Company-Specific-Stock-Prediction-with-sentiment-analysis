import pandas as pd

# ========================
# File paths
# ========================
sentiment_file = r"C:\Rishika\SPP\hdfc bank news\hdfc_sentiment_news.csv"
stock_file = r"C:\Rishika\SPP\data\HDFCBANK_data.csv"
output_file = r"C:\Rishika\SPP\hdfc bank news\merged_hdfc_sentiment_stock.csv"

# ========================
# Load both datasets
# ========================
sent_df = pd.read_csv(sentiment_file)
stock_df = pd.read_csv(stock_file)

# ========================
# Clean and convert date columns
# ========================

# For sentiment file (already in YYYY-MM-DD HH:MM:SS)
sent_df['date'] = pd.to_datetime(sent_df['date'], errors='coerce')

# For stock file (like "08:07:25 29/04/2025 pm IST")
stock_df['date'] = (
    stock_df['date']
    .str.replace('am IST', '', regex=False)
    .str.replace('pm IST', '', regex=False)
    .str.strip()
)
stock_df['date'] = pd.to_datetime(stock_df['date'], format='%H:%M:%S %d/%m/%Y', errors='coerce')

# ========================
# Extract only the date part for merging
# ========================
sent_df['date_only'] = sent_df['date'].dt.date
stock_df['date_only'] = stock_df['date'].dt.date

# ========================
# Merge on date_only
# ========================
merged_df = pd.merge(sent_df, stock_df, on='date_only', how='inner', suffixes=('_sent', '_stock'))

# ========================
# Save the merged dataset
# ========================
merged_df.to_csv(output_file, index=False)
print(f"✅ Merging complete. Saved to: {output_file}")

# ========================
# Optional: Show a sample
# ========================
print("\n📊 Merged Data Preview:")
print(merged_df.head(10))
