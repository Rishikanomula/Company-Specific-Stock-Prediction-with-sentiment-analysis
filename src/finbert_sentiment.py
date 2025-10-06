# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import pandas as pd
# from tqdm import tqdm

# # Load FinBERT model and tokenizer
# print("🔹 Loading FinBERT model...")
# model_name = "ProsusAI/finbert"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# # Load your news data (change path if needed)
# df = pd.read_csv("../data/live_news.csv")

# # Ensure there’s a text column
# if 'headline' not in df.columns and 'title' in df.columns:
#     df.rename(columns={'title': 'headline'}, inplace=True)

# print(f"📰 Analyzing {len(df)} news headlines...")

# sentiments = []
# for text in tqdm(df['headline'], desc="FinBERT Sentiment Analysis"):
#     try:
#         inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#         outputs = model(**inputs)
#         probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#         # FinBERT: [0]=negative, [1]=neutral, [2]=positive
#         score = probs[0][2].item() - probs[0][0].item()  # positive - negative
#         sentiments.append(score)
#     except Exception as e:
#         sentiments.append(0)
#         print(f"⚠️ Error processing: {text[:50]}... ({e})")

# # Add to DataFrame
# df['finbert_sentiment'] = sentiments

# # Save the output
# output_path = r"C:\Rishika\SPP\data\live_news_with_finbert.csv"
# df.to_csv(output_path, index=False)
# print(f"\n✅ FinBERT sentiment analysis complete!")
# print(f"📁 Saved results to: {output_path}")



from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

# Load FinBERT model and tokenizer
print("🔹 Loading FinBERT model...")
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load combined news data
news_file = "../data/live_news_with_company.csv"
df = pd.read_csv(news_file)

# Ensure columns exist
if 'headline' not in df.columns and 'title' in df.columns:
    df.rename(columns={'title': 'headline'}, inplace=True)

# Prepare sentiment list
sentiments = []

print(f"📰 Analyzing {len(df)} news headlines for all companies...")
for text in tqdm(df['headline'], desc="FinBERT Sentiment Analysis"):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # FinBERT: [0]=negative, [1]=neutral, [2]=positive
        score = probs[0][2].item() - probs[0][0].item()  # positive - negative
        sentiments.append(score)
    except Exception as e:
        sentiments.append(0)
        print(f"⚠️ Error processing: {text[:50]}... ({e})")

# Add sentiment to DataFrame
df['finbert_sentiment'] = sentiments

# Save output
output_file = r"C:\Rishika\SPP\data\live_news_with_finbert.csv"
df.to_csv(output_file, index=False)

print(f"\n✅ FinBERT sentiment analysis complete!")
print(f"📁 Saved results to: {output_file}")
