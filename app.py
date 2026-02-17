import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import os

st.set_page_config(page_title="VayuSutra AQI Intelligence", layout="wide")

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("data/processed_aqi.csv")
df['Date'] = pd.to_datetime(df['Date'])

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("⚙️ Control Panel")

cities = df['City'].unique()
city = st.sidebar.selectbox("Select City", cities)

mode = st.sidebar.radio(
    "Prediction Mode",
    ["Historical Forecast", "Manual Input Simulation"]
)

# ----------------------------
# CITY DATA (DEFINE ONCE)
# ----------------------------
full_city_df = df[df['City'] == city].sort_values("Date")

if len(full_city_df) < 7:
    st.error("Not enough data.")
    st.stop()

# Used for graphs (last 60 days)
city_df = full_city_df.tail(60)

latest_row = full_city_df.iloc[-1]
last_7 = full_city_df.tail(7)
next_date = latest_row['Date'] + timedelta(days=1)

# ----------------------------
# LOAD MODEL
# ----------------------------
model_path = f"models/city_models/xgb_{city}.pkl"

if not os.path.exists(model_path):
    st.error("Model not found.")
    st.stop()

model = joblib.load(model_path)

# ----------------------------
# HEADER
# ----------------------------
st.title("🌬️ VayuSutra - AQI Intelligence System")
st.markdown(f"### 📍 City: {city}")

# =====================================================
# HISTORICAL FORECAST MODE
# =====================================================
if mode == "Historical Forecast":

    st.subheader("📊 Historical AQI (Last 60 Days)")

    fig1 = px.line(
        city_df,
        x="Date",
        y="AQI",
        markers=True
    )

    fig1.update_layout(height=350, hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    # Forecast calculation
    forecast_values = []
    forecast_dates = []
    temp_df = full_city_df.copy()

    for i in range(7):
        latest = temp_df.iloc[-1]
        last7 = temp_df.tail(7)
        future_date = latest['Date'] + timedelta(days=1)

        input_data = {
            'PM2.5': latest['PM2.5'],
            'PM10': latest['PM10'],
            'NO2': latest['NO2'],
            'SO2': latest['SO2'],
            'CO': latest['CO'],
            'Year': future_date.year,
            'Month': future_date.month,
            'Day': future_date.day,
            'DayOfWeek': future_date.weekday(),
            'AQI_lag1': latest['AQI'],
            'AQI_lag2': temp_df.iloc[-2]['AQI'],
            'AQI_lag3': temp_df.iloc[-3]['AQI'],
            'AQI_7day_avg': last7['AQI'].mean()
        }

        pred = model.predict(pd.DataFrame([input_data]))[0]

        forecast_values.append(pred)
        forecast_dates.append(future_date)

        new_row = latest.copy()
        new_row['Date'] = future_date
        new_row['AQI'] = pred
        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])])

    st.subheader("📈 7-Day Forecast")

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=city_df["Date"],
        y=city_df["AQI"],
        mode="lines+markers",
        name="Actual"
    ))

    fig2.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode="lines+markers",
        name="Forecast"
    ))

    fig2.update_layout(height=350, hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# MANUAL SIMULATION MODE
# =====================================================
else:

    st.subheader("🎛️ Manual AQI Simulation")

    col1, col2 = st.columns(2)

    with col1:
        pm25 = st.slider("PM2.5", 0.0, 500.0, float(latest_row['PM2.5']))
        pm10 = st.slider("PM10", 0.0, 600.0, float(latest_row['PM10']))
        no2 = st.slider("NO2", 0.0, 300.0, float(latest_row['NO2']))

    with col2:
        so2 = st.slider("SO2", 0.0, 300.0, float(latest_row['SO2']))
        co = st.slider("CO", 0.0, 20.0, float(latest_row['CO']))

    input_data = {
        'PM2.5': pm25,
        'PM10': pm10,
        'NO2': no2,
        'SO2': so2,
        'CO': co,
        'Year': next_date.year,
        'Month': next_date.month,
        'Day': next_date.day,
        'DayOfWeek': next_date.weekday(),
        'AQI_lag1': latest_row['AQI'],
        'AQI_lag2': full_city_df.iloc[-2]['AQI'],
        'AQI_lag3': full_city_df.iloc[-3]['AQI'],
        'AQI_7day_avg': last_7['AQI'].mean()
    }

    prediction = model.predict(pd.DataFrame([input_data]))[0]

    st.metric("Predicted AQI", round(prediction, 2))
