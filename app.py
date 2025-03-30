import streamlit as st
import pandas as pd
import numpy as np
import os
from fbprophet import Prophet
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="AI Revenue Forecasting", page_icon="📈", layout="wide")
st.title("📊 AI-Driven Revenue Forecasting")

# File Upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())
    
    if "Date" in df.columns and "Revenue" in df.columns:
        df.rename(columns={"Date": "ds", "Revenue": "y"}, inplace=True)
        df["ds"] = pd.to_datetime(df["ds"], errors='coerce')
        df.dropna(subset=["ds", "y"], inplace=True)
        
        # Prophet Model
        model = Prophet()
        model.fit(df)
        
        # Future Predictions
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        # Plot Forecast
        st.write("### Forecast Results")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)
        
        # Plot Components
        st.write("### Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
    else:
        st.error("The uploaded file must contain 'Date' and 'Revenue' columns.")
