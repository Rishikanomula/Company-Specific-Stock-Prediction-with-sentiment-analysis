import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(page_title="Stock Price Prediction (LSTM)", layout="wide")

st.title("📈 Stock Price Prediction using LSTM")
st.write("Predict next-day closing price for Indian companies using a fine-tuned LSTM deep learning model.")

@st.cache_resource
def load_resources():
    model = load_model(r"C:\Rishika\SPP\HDFC\lstm_model.h5", compile=False)

    scaler = joblib.load(r"C:\Rishika\SPP\HDFC\scaler.pkl")
    return model, scaler

model, scaler = load_resources()

data = pd.read_csv(r"C:\Rishika\SPP\HDFC\merged_hdfc_stock_sentiment.csv")

if 'Company' in data.columns:
    companies = data['Company'].unique()
    selected_company = st.selectbox("Select a Company:", companies)
    company_data = data[data['Company'] == selected_company]
else:
    company_data = data
    selected_company = "HDFC Bank"

st.subheader(f"📊 Historical Stock Data – {selected_company}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=company_data['Date'], y=company_data['Close'], mode='lines', name='Close Price'))
fig.update_layout(xaxis_title='Date', yaxis_title='Price', template='plotly_white')
st.plotly_chart(fig, use_container_width=True)

st.subheader("💰 Predict Next-Day Closing Price")
window_size = 30
recent_data = company_data[['Close', 'compound_score']].tail(window_size).values

recent_scaled = scaler.transform(recent_data)
X_input = np.expand_dims(recent_scaled, axis=0)
predicted_price_scaled = model.predict(X_input)
dummy = np.zeros((1, recent_data.shape[1]))
dummy[0, 0] = predicted_price_scaled[0, 0]
predicted_price = scaler.inverse_transform(dummy)[0, 0]
st.metric(label="Predicted Next-Day Price", value=f"₹{predicted_price:.2f}")

st.subheader("📉 Model Performance Visualizations")
col1, col2 = st.columns(2)
with col1:
    st.image(r"C:\Rishika\SPP\app\Figure_1.png", caption="LSTM without Sentiment", use_container_width=True)
with col2:
    st.image(r"C:\Rishika\SPP\app\Figure_2.png", caption="LSTM with Sentiment", use_container_width=True)
st.image(r"C:\Rishika\SPP\app\lstm_fined_tuned_modelloss.png", caption="Model Training Loss Curve", use_container_width=True)
