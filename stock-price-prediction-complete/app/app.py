import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from src.features import add_features, recursive_predict_next_n
from src.model import train_eval_time_split

st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction (Random Forest)")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    start = st.date_input("Start date", value=date(2015,1,1))
    end = st.date_input("End date", value=date.today())
    horizon = st.number_input("Forecast horizon (days)", min_value=1, max_value=30, value=5)
    save_model = st.checkbox("Save model", value=False)
    run_btn = st.button("Train & Predict")

status = st.empty()

@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.index.name = "Date"
    return df

def plot_history(df):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, df['Close'])
    ax.set_title("Close Price History")
    st.pyplot(fig)

def plot_future(df_hist, preds):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_hist.index, df_hist['Close'], label="History")
    ax.plot(preds.index, preds['PredictedClose'], label="Forecast")
    ax.legend()
    st.pyplot(fig)

if run_btn:
    status.info("Downloading data…")
    data = load_data(ticker, start, end)
    if data.empty:
        st.error("No data found.")
        st.stop()
    st.subheader("History")
    plot_history(data)
    status.info("Engineering features…")
    df_feat, feature_cols, target_col = add_features(data)
    status.info("Training…")
    model, report, _ = train_eval_time_split(df_feat, feature_cols, target_col)
    st.subheader("Metrics")
    st.write(report)
    status.info("Forecasting…")
    preds = recursive_predict_next_n(data, model, feature_cols, int(horizon))
    st.subheader("Forecast")
    st.dataframe(preds)
    plot_future(data, preds)
    if save_model:
        Path("models").mkdir(exist_ok=True)
        out_path = Path("models") / f"{ticker}_rf.pkl"
        joblib.dump({"model":model,"features":feature_cols}, out_path)
        st.success(f"Saved model to {out_path}")
    status.success("Done!")
else:
    st.info("Configure settings in the sidebar and click Train & Predict")
