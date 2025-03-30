import streamlit as st
import pandas as pd
import numpy as np
import os
from fbprophet import Prophet
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="AI Forecasting with Prophet", page_icon="📊", layout="wide")
st.title("📈 AI Forecasting with Prophet")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file with 'Date' and 'Revenue' columns", type=["xls", "xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("### Raw Data")
    st.write(df.head())
    
    # Ensure correct column names
    df.rename(columns={"Date": "ds", "Revenue": "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])
    
    # Prophet Model
    model = Prophet()
    model.fit(df)
    
    # Future Dataframe
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Plot forecast
    st.write("### Forecast Results")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)
    
    # Display Forecast Data
    st.write("### Forecast Data")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    # Additional insights
    st.write("### Trend and Seasonality")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
