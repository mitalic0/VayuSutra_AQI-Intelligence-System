import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model

print("Loading validation data...")

val_df = pd.read_csv("data/validation.csv")

results = []

# -----------------------------
# XGBOOST VALIDATION
# -----------------------------

import os

cities = val_df['City'].unique()

xgb_results = []

for city in cities:
    city_df = val_df[val_df['City'] == city]

    if len(city_df) < 10:
        continue

    X_val = city_df.drop(['AQI','Date','City'], axis=1)
    y_val = city_df['AQI']

    model_path = f"models/city_models/xgb_{city}.pkl"

    if not os.path.exists(model_path):
        continue

    model = joblib.load(model_path)

    preds = model.predict(X_val)

    xgb_results.append([
        city,
        np.sqrt(mean_squared_error(y_val, preds)),
        mean_absolute_error(y_val, preds),
        r2_score(y_val, preds)
    ])

xgb_results_df = pd.DataFrame(
    xgb_results,
    columns=["City","RMSE","MAE","R2"]
)

print("\nPer-City XGBoost Results:")
print(xgb_results_df)

xgb_results_df.to_csv("results/xgb_per_city_results.csv", index=False)

# -----------------------------
# LSTM VALIDATION
# -----------------------------

print("Validating LSTM...")

lstm_model = load_model("models/lstm_model.h5", compile=False)
scaler = joblib.load("models/lstm_scaler.pkl")

scaled = scaler.transform(val_df[['AQI']].values)

X_val_lstm, y_val_lstm = [], []
seq_len = 30

for i in range(seq_len, len(scaled)):
    X_val_lstm.append(scaled[i-seq_len:i])
    y_val_lstm.append(scaled[i])

X_val_lstm = np.array(X_val_lstm)
y_val_lstm = np.array(y_val_lstm)

lstm_pred = lstm_model.predict(X_val_lstm)

lstm_pred_inv = scaler.inverse_transform(lstm_pred)
y_val_inv = scaler.inverse_transform(y_val_lstm)

results.append([
    "LSTM",
    np.sqrt(mean_squared_error(y_val_inv, lstm_pred_inv)),
    mean_absolute_error(y_val_inv, lstm_pred_inv),
    r2_score(y_val_inv, lstm_pred_inv)
])

# -----------------------------
# SAVE RESULTS
# -----------------------------

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R2"])
results_df.to_csv("results/model_comparison.csv", index=False)

print("\nValidation Results:")
print(results_df)
print("\nResults saved to results/model_comparison.csv")
