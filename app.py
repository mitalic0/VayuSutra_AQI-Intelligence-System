import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="VayuSutra AQI Intelligence", layout="wide")

# =====================================================
# CUSTOM ENVIRONMENTAL THEME
# =====================================================
st.markdown("""
<style>

/* Sky + Land Gradient Background */
.stApp {
    background: linear-gradient(to bottom, #87CEEB 0%, #f0f8ff 40%, #d4f4dd 100%);
}

/* Main content container */
.block-container {
    background-color: rgba(255, 255, 255, 0.92);
    padding: 2rem;
    border-radius: 20px;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #e0f7fa, #c8e6c9);
}

/* Header styling */
h1, h2, h3 {
    color: #1b5e20;
}

/* Metric styling */
[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.95);
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("data/processed_aqi.csv")
df['Date'] = pd.to_datetime(df['Date'])

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
st.sidebar.title("🌬️ VayuSutra Control Panel")

cities = df['City'].unique()
city = st.sidebar.selectbox("📍 Select City", cities)

min_date = df['Date'].min()
max_date = df['Date'].max()

start_date = st.sidebar.date_input(
    "📅 Start Date",
    value=max_date - pd.Timedelta(days=90),
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "📅 End Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

page = st.sidebar.radio(
    "📂 Navigate",
    ["📊 Analytics", "🔮 Forecast", "🎛 Simulation Lab", "🧠 AI Insights", "🌍 City Comparison"]
)

# =====================================================
# DATA PREPARATION
# =====================================================
full_city_df = df[df['City'] == city].sort_values("Date")

if len(full_city_df) < 7:
    st.error("Not enough data for forecasting.")
    st.stop()

filtered_city_df = full_city_df[
    (full_city_df['Date'] >= pd.to_datetime(start_date)) &
    (full_city_df['Date'] <= pd.to_datetime(end_date))
]

latest_row = full_city_df.iloc[-1]
last_7 = full_city_df.tail(7)
next_date = latest_row['Date'] + timedelta(days=1)

# =====================================================
# LOAD MODEL
# =====================================================
model_path = f"models/city_models/xgb_{city}.pkl"

if not os.path.exists(model_path):
    st.error("Model not found. Train models first.")
    st.stop()

model = joblib.load(model_path)

# =====================================================
# HEADER
# =====================================================
st.title("🌬️ VayuSutra - AQI Intelligence System")
st.markdown(f"### 📍 Selected City: {city}")

# =====================================================
# ANALYTICS PAGE
# =====================================================
if page == "📊 Analytics":

    st.subheader("Historical AQI Trend")

    if filtered_city_df.empty:
        st.warning("No data for selected range.")
    else:
        fig = px.line(
            filtered_city_df,
            x="Date",
            y="AQI",
            markers=True
        )
        fig.update_layout(hovermode="x unified", height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        pollutant = st.selectbox(
            "Explore Pollutant Trend",
            ["PM2.5", "PM10", "NO2", "SO2", "CO"]
        )

        fig2 = px.line(
            filtered_city_df,
            x="Date",
            y=pollutant
        )
        fig2.update_layout(hovermode="x unified", height=450)
        st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# FORECAST PAGE
# =====================================================
elif page == "🔮 Forecast":

    st.subheader("7-Day AQI Forecast")

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

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode="lines+markers",
        name="Forecast AQI"
    ))
    fig3.update_layout(hovermode="x unified", height=450)
    st.plotly_chart(fig3, use_container_width=True)

    st.download_button(
        "📥 Download Forecast CSV",
        data=pd.DataFrame({
            "Date": forecast_dates,
            "Forecast_AQI": forecast_values
        }).to_csv(index=False),
        file_name="forecast.csv"
    )

# =====================================================
# SIMULATION LAB
# =====================================================
elif page == "🎛 Simulation Lab":

    st.subheader("AQI Simulation Lab")

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

    fig4 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Predicted AQI"},
        gauge={
            'axis': {'range': [0, 500]},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 100], 'color': "yellow"},
                {'range': [100, 200], 'color': "orange"},
                {'range': [200, 500], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig4, use_container_width=True)

# =====================================================
# CITY COMPARISON
# =====================================================
elif page == "🌍 City Comparison":

    st.subheader("Compare Cities")

    selected_cities = st.multiselect(
        "Select Cities",
        df['City'].unique(),
        default=[city]
    )

    filtered_df = df[
        (df['City'].isin(selected_cities)) &
        (df['Date'] >= pd.to_datetime(start_date)) &
        (df['Date'] <= pd.to_datetime(end_date))
    ]

    if filtered_df.empty:
        st.warning("No data for selected range.")
    else:
        fig = px.line(
            filtered_df,
            x="Date",
            y="AQI",
            color="City",
            markers=True
        )
        fig.update_layout(hovermode="x unified", height=450)
        st.plotly_chart(fig, use_container_width=True)

        summary = (
            filtered_df.groupby("City")["AQI"]
            .agg(["mean", "max", "min"])
            .reset_index()
        )

        summary.columns = ["City", "Average AQI", "Max AQI", "Min AQI"]

        st.dataframe(summary)
