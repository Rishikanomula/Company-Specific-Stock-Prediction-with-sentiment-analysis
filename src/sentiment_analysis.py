import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download lexicon if not already downloaded
nltk.download('vader_lexicon')

# Load the news data
input_file = r"C:\Rishika\SPP\data\live_news.csv"
output_file = r"C:\Rishika\SPP\data\live_news_with_sentiment_1.csv"

df = pd.read_csv(input_file)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to classify sentiment
def get_sentiment(text):
    score = sia.polarity_scores(str(text))['compound']
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
df['sentiment'] = df['headline'].apply(get_sentiment)

# Save the results
df.to_csv(output_file, index=False)
print(f"✅ Sentiment analysis complete. Saved to {output_file}")

# Optional: see sentiment counts
print("\n🧾 Sentiment Summary:")
print(df['sentiment'].value_counts())
