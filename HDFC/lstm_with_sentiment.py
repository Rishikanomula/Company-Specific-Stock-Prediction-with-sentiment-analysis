# HDFC Stock Price Prediction using LSTMwith Sentiment Analysis
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# -------------------------------
# Load the merged dataset
# -------------------------------
data_path = r"C:\Rishika\SPP\HDFC\merged_hdfc_stock_sentiment.csv"  
df = pd.read_csv(data_path)

# Ensure 'Date' is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date (important for time-series)
df = df.sort_values('Date')

# -------------------------------
# Encode Sentiment & Scale Features
# -------------------------------

# Encode sentiment to numeric (Neutral=1, Positive=2, Negative=0)
encoder = LabelEncoder()
df['Sentiment_encoded'] = encoder.fit_transform(df['sentiment'])

# Select features
features = ['Open', 'High', 'Low', 'Volume', 'Sentiment_encoded']
target = 'Close'

# Normalize the data for LSTM stability
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features + [target]])

scaled_df = pd.DataFrame(scaled_data, columns=features + [target])

# -------------------------------
# Create Time-Series Sequences
# -------------------------------
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, :-1])  # all features except target
        y.append(data[i, -1])             # target (Close)
    return np.array(X), np.array(y)

SEQ_LEN = 5  # lookback window (past 5 days)
X, y = create_sequences(scaled_data, SEQ_LEN)

# Split into train and test sets (80-20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------
# Build the LSTM Model
# -------------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(features))),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # output layer for regression
])

model.compile(optimizer='adam', loss='mse')

# -------------------------------
# Train the Model
# -------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------
# Evaluate Model Performance
# -------------------------------
y_pred = model.predict(X_test)

# Reverse scaling for actual price comparison
scale_close = scaler.scale_[-1]
min_close = scaler.min_[-1]
y_test_actual = y_test / scale_close - min_close / scale_close
y_pred_actual = y_pred / scale_close - min_close / scale_close

# ✅ Save values for evaluation
np.save('y_true.npy', y_test_actual)                 # Actual stock prices
np.save('y_pred_sentiment.npy', y_pred_actual)  

mae = mean_absolute_error(y_test_actual, y_pred_actual)
mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred_actual)

print("\n📊 LSTM Model Performance (with Sentiment):")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# -------------------------------
# Visualize Predictions
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test_actual, label='Actual Prices', color='blue')
plt.plot(y_pred_actual, label='Predicted Prices', color='orange')
plt.title('HDFC Stock Price Prediction (LSTM with Sentiment)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
