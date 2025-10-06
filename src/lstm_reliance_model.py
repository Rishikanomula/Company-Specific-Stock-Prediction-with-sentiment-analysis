import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Load merged dataset
# -------------------------------
file_path = r"C:\Rishika\SPP\data\RELIANCE_merged.csv"
df = pd.read_csv(file_path)

# Ensure correct types
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Keep only last 6 months for meaningful sentiment coverage
recent_df = df[df['Date'] >= (df['Date'].max() - pd.Timedelta(days=180))].copy()

# Fill missing sentiment with 0
recent_df['finbert_sentiment'] = recent_df['finbert_sentiment'].fillna(0.0)

# -------------------------------
# 2️⃣ Select features
# -------------------------------
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'finbert_sentiment']
target = 'Close'
data = recent_df[features]

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# -------------------------------
# 3️⃣ Create sequences
# -------------------------------
def create_sequences(dataset, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(dataset)):
        X.append(dataset[i - time_steps:i, :-1])  # all features except target
        y.append(dataset[i, 3])  # 'Close' is 4th column in features
    return np.array(X), np.array(y)

time_steps = 60
X, y = create_sequences(scaled_data, time_steps)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# -------------------------------
# 4️⃣ LSTM Model
# -------------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
print("\n🧠 Training LSTM with sentiment...")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# -------------------------------
# 5️⃣ Predict & inverse scale
# -------------------------------
predicted = model.predict(X_test)
close_scaler = MinMaxScaler()
close_scaler.min_, close_scaler.scale_ = scaler.min_[3], scaler.scale_[3]  # only for Close column
predicted_prices = close_scaler.inverse_transform(predicted.reshape(-1,1))
actual_prices = close_scaler.inverse_transform(y_test.reshape(-1,1))

# -------------------------------
# 6️⃣ Evaluate
# -------------------------------
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
r2 = r2_score(actual_prices, predicted_prices)
print(f"\nTest RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

# -------------------------------
# 7️⃣ Plot results
# -------------------------------
plt.figure(figsize=(10,6))
plt.plot(actual_prices, label='Actual Close Price')
plt.plot(predicted_prices, label='Predicted Close Price')
plt.title('Reliance Stock Price Prediction (LSTM + Sentiment)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
