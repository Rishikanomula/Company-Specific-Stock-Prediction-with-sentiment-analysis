import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========================
# Load merged dataset
# ========================
file_path = r"C:\Rishika\SPP\HDFC\merged_hdfc_stock_sentiment.csv"
df = pd.read_csv(file_path)

# ========================
# Preprocessing
# ========================
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Encode sentiment if textual (Positive/Negative/Neutral)
if 'sentiment' in df.columns and df['sentiment'].dtype == 'object':
    df['sentiment_encoded'] = df['sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
else:
    df['sentiment_encoded'] = df['compound_score'] if 'compound_score' in df.columns else 0

# Shift Close to create next-day prediction target
df['Next_Close'] = df['Close'].shift(-1)
df = df.dropna(subset=['Next_Close'])

# ========================
# Feature selection
# ========================
features = ['Open', 'High', 'Low', 'Volume', 'sentiment_encoded']
target = 'Next_Close'

X = df[features]
y = df[target]

# ========================
# Train-test split (time-based)
# ========================
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# ========================
# Model training
# ========================
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ========================
# Predictions
# ========================
y_pred = model.predict(X_test)

# ========================
# Evaluation metrics
# ========================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance (with Sentiment):")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ========================
# Optional: Baseline model (without sentiment)
# ========================
X_base = df[['Open', 'High', 'Low', 'Volume']]
y_base = y

X_train_b, X_test_b = X_base[:split_index], X_base[split_index:]
y_train_b, y_test_b = y_base[:split_index], y_base[split_index:]

base_model = RandomForestRegressor(n_estimators=200, random_state=42)
base_model.fit(X_train_b, y_train_b)
y_pred_b = base_model.predict(X_test_b)

mae_b = mean_absolute_error(y_test_b, y_pred_b)
rmse_b = np.sqrt(mean_squared_error(y_test_b, y_pred_b))
r2_b = r2_score(y_test_b, y_pred_b)

print("\n📉 Baseline Model (without Sentiment):")
print(f"MAE  : {mae_b:.4f}")
print(f"RMSE : {rmse_b:.4f}")
print(f"R²   : {r2_b:.4f}")

# ========================
# Comparison summary
# ========================
print("\n🧾 COMPARISON SUMMARY")
comparison = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R2'],
    'Without Sentiment': [mae_b, rmse_b, r2_b],
    'With Sentiment': [mae, rmse, r2]
})
print(comparison)
