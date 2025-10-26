import pandas as pd

data_path = r"C:\Rishika\SPP\hdfc_merged_sentiment.csv"
df = pd.read_csv(data_path)

print(df.columns)
print(df.head())
