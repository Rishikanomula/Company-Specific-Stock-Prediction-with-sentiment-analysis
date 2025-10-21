import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ======================
# 1️⃣ Load and Prepare Data
# ======================
file_path = r"C:\Rishika\SPP\data\HDFCBANK_data.csv"
df = pd.read_csv(file_path)

# Convert 'Date' to datetime and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Use only the 'Close' price
data = df[['Close']].values

# Train-test split (before scaling)
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# ======================
# 2️⃣ Normalize Data
# ======================
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# ======================
# 3️⃣ Create Sequences
# ======================
def create_sequences(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i + time_step, 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_sequences(train_scaled, time_step)
X_test, y_test = create_sequences(test_scaled, time_step)

# Reshape for LSTM (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# ======================
# 4️⃣ Build LSTM Model
# ======================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.summary()

# ======================
# 5️⃣ Train Model
# ======================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# ======================
# 6️⃣ Predict and Inverse Transform
# ======================
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse scaling
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_true = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))

# ======================
# 7️⃣ Evaluate Model
# ======================
train_rmse = np.sqrt(mean_squared_error(y_train_true, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_true, test_predict))
r2 = r2_score(y_test_true, test_predict)

print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# ======================
# 8️⃣ Plot Results
# ======================
plt.figure(figsize=(10,6))
plt.plot(y_test_true, label="Actual Price", color='blue')
plt.plot(test_predict, label="Predicted Price", color='red')
plt.title("HDFC Bank Stock Price Prediction (LSTM)")
plt.xlabel("Time")
plt.ylabel("Price (₹)")
plt.legend()
plt.show()
