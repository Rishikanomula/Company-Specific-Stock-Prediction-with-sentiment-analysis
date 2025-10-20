import pandas as pd

# Load the full news dataset (replace 'news_dataset.csv' with your dataset's filename)
df = pd.read_csv('news_dataset.csv')

# List of target companies
companies = ["HDFC Bank", "TCS", "Infosys", "Reliance"]

# Loop and filter for each company
for company in companies:
    filtered_df = df[df['Text'].str.contains(company, case=False, na=False)]
    filtered_df.to_csv(f"{company.replace(' ', '')}_news_filtered.csv", index=False)
