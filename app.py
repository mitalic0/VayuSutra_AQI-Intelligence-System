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
# ENVIRONMENTAL THEME
# =====================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #87CEEB 0%, #f0f8ff 40%, #d4f4dd 100%);
}
.block-container {
    background-color: rgba(255,255,255,0.92);
    padding: 2rem;
    border-radius: 20px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #e0f7fa, #c8e6c9);
}
h1, h2, h3 {
    color: #1b5e20;
}
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
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed_aqi.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🌬️ VayuSutra Control Panel")

cities = sorted(df['City'].unique())
city = st.sidebar.selectbox("📍 Select City", cities)

city_df = df[df['City'] == city].sort_values("Date")

min_date = city_df['Date'].min()
max_date = city_df['Date'].max()

start_date = st.sidebar.date_input(
    "📅 Start Date",
    value=max_date - pd.Timedelta(days=60),
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "📅 End Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

page = st.sidebar.radio(
    "📂 Navigate",
    ["📊 Analytics", "🔮 Forecast", "🎛 Simulation Lab", "🌍 City Comparison"]
)

# =====================================================
# FILTER DATA
# =====================================================
filtered_df = city_df[
    (city_df['Date'] >= pd.to_datetime(start_date)) &
    (city_df['Date'] <= pd.to_datetime(end_date))
]

if filtered_df.empty:
    filtered_df = city_df.tail(60)
    st.sidebar.warning("Showing latest 60 days (no data in range).")

latest_row = city_df.iloc[-1]
last_7 = city_df.tail(7)
next_date = latest_row['Date'] + timedelta(days=1)

# =====================================================
# AUTO LOAD OR TRAIN MODEL (DEPLOY SAFE)
# =====================================================
@st.cache_resource
def load_or_train_model(city_name):

    model_path = f"models/city_models/xgb_{city_name}.pkl"

    if os.path.exists(model_path):
        return joblib.load(model_path)

    else:
        st.info("Training model for first-time deployment...")

        from models.train_xgboost import train_model_for_city
        model = train_model_for_city(city_name)

        return model

model = load_or_train_model(city)

# =====================================================
# HEADER
# =====================================================
st.title("🌬️ VayuSutra - AQI Intelligence System")
st.markdown(f"### 📍 Selected City: {city}")
st.caption("Use the sidebar to explore different analytics modules.")

# =====================================================
# ANALYTICS
# =====================================================
if page == "📊 Analytics":

    st.subheader("Historical AQI Trend")

    fig = px.line(
        filtered_df,
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
        filtered_df,
        x="Date",
        y=pollutant
    )
    fig2.update_layout(hovermode="x unified", height=450)
    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# FORECAST
# =====================================================
elif page == "🔮 Forecast":

    st.subheader("7-Day AQI Forecast")

    forecast_vals = []
    forecast_dates = []
    temp_df = city_df.copy()

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

        forecast_vals.append(pred)
        forecast_dates.append(future_date)

        new_row = latest.copy()
        new_row['Date'] = future_date
        new_row['AQI'] = pred
        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])])

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_vals,
        mode="lines+markers",
        name="Forecast AQI"
    ))

    fig3.update_layout(hovermode="x unified", height=450)
    st.plotly_chart(fig3, use_container_width=True)

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
        'AQI_lag2': city_df.iloc[-2]['AQI'],
        'AQI_lag3': city_df.iloc[-3]['AQI'],
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
        cities,
        default=[city]
    )

    comp_df = df[
        (df['City'].isin(selected_cities)) &
        (df['Date'] >= pd.to_datetime(start_date)) &
        (df['Date'] <= pd.to_datetime(end_date))
    ]

    if not comp_df.empty:

        fig = px.line(
            comp_df,
            x="Date",
            y="AQI",
            color="City",
            markers=True
        )
        fig.update_layout(hovermode="x unified", height=450)
        st.plotly_chart(fig, use_container_width=True)

        summary = comp_df.groupby("City")["AQI"].agg(["mean","max","min"]).reset_index()
        summary.columns = ["City","Average AQI","Max AQI","Min AQI"]

        st.dataframe(summary)

    else:
        st.warning("No data for selected range.")
