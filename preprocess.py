import pandas as pd

print("Loading dataset...")

df = pd.read_csv("data/city_day.csv")
df.columns = df.columns.str.strip()

df['Date'] = pd.to_datetime(df['Datetime'])

# Remove missing AQI
df = df[df['AQI'].notna()]

# Remove extreme AQI
df = df[df['AQI'] < 600]

df.sort_values(["City", "Date"], inplace=True)

# Time features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek

# -----------------------------
# LAG FEATURES (per city)
# -----------------------------
df['AQI_lag1'] = df.groupby('City')['AQI'].shift(1)
df['AQI_lag2'] = df.groupby('City')['AQI'].shift(2)
df['AQI_lag3'] = df.groupby('City')['AQI'].shift(3)

# -----------------------------
# 7-Day Rolling Mean (per city)
# -----------------------------
df['AQI_7day_avg'] = (
    df.groupby('City')['AQI']
      .rolling(7)
      .mean()
      .reset_index(level=0, drop=True)
)

# Drop rows created by lag/rolling
df.dropna(inplace=True)

features = [
    'PM2.5','PM10','NO2','SO2','CO',
    'Year','Month','Day','DayOfWeek',
    'AQI_lag1','AQI_lag2','AQI_lag3',
    'AQI_7day_avg'
]

df = df[['Date','City'] + features + ['AQI']]

df.to_csv("data/processed_aqi.csv", index=False)

print("Preprocessing with lag + rolling features completed.")
