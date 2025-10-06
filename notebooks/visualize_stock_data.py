import pandas as pd
import matplotlib.pyplot as plt

# Read one company’s data
df = pd.read_csv(r'C:\Rishika\SPP\data\RELIANCE_data.csv', skiprows=[1])

# Convert columns to numeric where applicable
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# # Show first few rows
# print("Sample Data:")
# print(df.head())

# Basic info
print("\nData Info:")
print(df.info())

# Plot Close price over time
plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Close'], label='Close Price', linewidth=2)
plt.title('Reliance Stock Price (2020–Present)')
plt.xlabel('Date')
plt.ylabel('Close Price (INR)')
plt.legend()
plt.grid(True)
plt.show()
