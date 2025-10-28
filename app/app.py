import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Stock Price Prediction (LSTM + Sentiment)", layout="wide")

st.title("📈 Stock Price Prediction using LSTM and Sentiment Analysis")
st.write("Predict next-day closing price for Indian companies using deep learning and sentiment analysis.")

# -------------------------
# LOAD MODEL AND SCALER
# -------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_resources()

# -------------------------
# LOAD DATA
# -------------------------
data = pd.read_csv("sentiment_data.csv")
companies = data['Company'].unique()
selected_company = st.selectbox("Select a Company:", companies)

company_data = data[data['Company'] == selected_company]

# -------------------------
# VISUALIZE HISTORICAL DATA
# -------------------------
st.subheader("📊 Historical Stock Data")
fig = go.Figure()
fig.add_trace(go.Scatter(x=company_data['Date'], y=company_data['Close'], mode='lines', name='Close Price'))
fig.update_layout(xaxis_title='Date', yaxis_title='Price', template='plotly_white')
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# SENTIMENT ANALYSIS
# -------------------------
st.subheader("📰 Sentiment Analysis on Latest News")
analyzer = SentimentIntensityAnalyzer()
user_news = st.text_area("Enter latest financial headline for this company:")
if st.button("Analyze Sentiment"):
    sentiment = analyzer.polarity_scores(user_news)
    st.write("**Sentiment Score:**", sentiment)
    st.write(f"🟢 Positive: {sentiment['pos']}, 🔴 Negative: {sentiment['neg']}, ⚪ Neutral: {sentiment['neu']}, 🧠 Compound: {sentiment['compound']}")

# -------------------------
# PRICE PREDICTION
# -------------------------
st.subheader("💰 Predict Next-Day Closing Price")

window_size = 30  # depends on your model training setup
recent_data = company_data[['Close', 'Sentiment']].tail(window_size).values
recent_scaled = scaler.transform(recent_data)
X_input = np.expand_dims(recent_scaled, axis=0)
predicted_price = model.predict(X_input)
predicted_price = scaler.inverse_transform([[predicted_price[0][0], 0]])[0][0]

st.metric(label="Predicted Next-Day Price", value=f"₹{predicted_price:.2f}")

# -------------------------
# COMPARISON PLOTS
# -------------------------
st.subheader("📉 Model Performance Visualizations")

col1, col2 = st.columns(2)
with col1:
    st.image("images/Figure_1.png", caption="LSTM without Sentiment", use_container_width=True)
with col2:
    st.image("images/Figure_2.png", caption="LSTM with Sentiment", use_container_width=True)

st.image("images/lstm_fined_tuned_modelloss.png", caption="Model Training Loss Curve", use_container_width=True)
