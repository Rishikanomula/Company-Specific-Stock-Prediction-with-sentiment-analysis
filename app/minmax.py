import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load your original dataset
data = pd.read_csv(r"C:\Rishika\SPP\HDFC\merged_hdfc_stock_sentiment.csv")

# Keep only the 5 features your model actually used
features = ['Close', 'High', 'Low', 'Open', 'Volume']

scaler = MinMaxScaler()
scaler.fit(data[features])

# Save the new scaler
joblib.dump(scaler, r"C:\Rishika\SPP\HDFC\scaler.pkl")

print("✅ New scaler saved for 5 features.")
