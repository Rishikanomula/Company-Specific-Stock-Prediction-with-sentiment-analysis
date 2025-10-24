import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# ✅ Download lexicon if not already downloaded
nltk.download('vader_lexicon')

# ✅ File paths
input_file = r"C:\Rishika\SPP\hdfc bank news\HDFC_news_21.csv"
output_file = r"C:\Rishika\SPP\hdfc bank news\hdfc_sentiment_news.csv"

# ✅ Load dataset
df = pd.read_csv(input_file)

# ✅ Clean and convert date column to proper datetime format
# Remove 'am IST' / 'pm IST' and convert to datetime
df['date'] = (
    df['date']
    .str.replace('am IST', '', regex=False)
    .str.replace('pm IST', '', regex=False)
    .str.strip()
)
df['date'] = pd.to_datetime(df['date'], format='%H:%M:%S %d/%m/%Y', errors='coerce')

# ✅ Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# ✅ Function to classify sentiment based on compound score
def get_sentiment(text):
    score = sia.polarity_scores(str(text))['compound']
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# ✅ Apply sentiment analysis
df['compound_score'] = df['headline'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
df['sentiment'] = df['headline'].apply(get_sentiment)

# ✅ Save results to new CSV
df.to_csv(output_file, index=False)
print(f"✅ Sentiment analysis complete. Saved to {output_file}")

# ✅ Optional: Display summary
print("\n🧾 Sentiment Summary:")
print(df['sentiment'].value_counts())

# ✅ (Optional) Show top few analyzed headlines
print("\n📊 Sample results:")
print(df[['headline', 'date', 'compound_score', 'sentiment']].head(10))
