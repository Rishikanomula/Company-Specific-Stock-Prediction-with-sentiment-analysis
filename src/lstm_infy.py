import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load dataset
file_path = r"C:\Rishika\SPP\data\INFY_data.csv"
df = pd.read_csv(file_path)

# Ensure Date column is datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Show few rows
# print(df.head())

data = df[['Close']].values

# Normalize between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_sequences(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_sequences(scaled_data, time_step)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))


train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


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


history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)


train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Reverse scaling
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_true = scaler.inverse_transform([y_train])
y_test_true = scaler.inverse_transform([y_test])


from sklearn.metrics import mean_squared_error, r2_score

train_rmse = np.sqrt(mean_squared_error(y_train_true[0], train_predict[:,0]))
test_rmse = np.sqrt(mean_squared_error(y_test_true[0], test_predict[:,0]))
r2 = r2_score(y_test_true[0], test_predict[:,0])

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
print(f"R² Score: {r2}")

plt.figure(figsize=(10,6))
plt.plot(y_test_true[0], label="Actual Price")
plt.plot(test_predict, label="Predicted Price")
plt.title("TCS Stock Price Prediction (LSTM)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
