# ======================================
# 📦 Import Required Libraries
# ======================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ======================================
# 🧾 Load Your Dataset
# ======================================
# Example: Merged dataset with sentiment column
# Columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment_Score']
data = pd.read_csv("HDFC_with_sentiment.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# ======================================
# ⚙️ Feature Selection
# ======================================
features = ['Open', 'High', 'Low', 'Volume', 'Sentiment_Score']
target = 'Close'

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(data[features + [target]])

scaled_df = pd.DataFrame(scaled_features, columns=features + [target])

# ======================================
# 🪄 Sequence Creation
# ======================================
SEQ_LEN = 10  # You can try 5, 10, 20

X, y = [], []
for i in range(SEQ_LEN, len(scaled_df)):
    X.append(scaled_df[features].iloc[i-SEQ_LEN:i].values)
    y.append(scaled_df[target].iloc[i])

X, y = np.array(X), np.array(y)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# ======================================
# 🧩 Train-Test Split
# ======================================
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# ======================================
# 🧠 Fine-Tuned LSTM Model Architecture
# ======================================
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, len(features))),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64, return_sequences=False),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(1)
])

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mse')

model.summary()

# ======================================
# ⏱️ Training with Callbacks
# ======================================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ======================================
# 📊 Plot Training History
# ======================================
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# ======================================
# 🔍 Prediction and Evaluation
# ======================================
y_pred = model.predict(X_test)

# Reverse scaling for accurate metrics
full_scaler = MinMaxScaler()
full_scaler.fit(data[[target]])
y_test_actual = full_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_actual = full_scaler.inverse_transform(y_pred)

mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
r2 = r2_score(y_test_actual, y_pred_actual)

print("📊 Fine-Tuned LSTM Model Performance (with Sentiment):")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ======================================
# 📈 Visualize Predicted vs Actual
# ======================================
plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label='Actual Prices', color='blue')
plt.plot(y_pred_actual, label='Predicted Prices', color='red')
plt.title('Stock Price Prediction (Fine-Tuned LSTM)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
