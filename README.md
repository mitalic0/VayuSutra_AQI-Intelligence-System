
# VayuSutra - Multi-City Enhanced AQI Forecasting System

## Dataset Required
Place Kaggle 'city_day.csv' in:
data/city_day.csv

## Execution Flow
1. python preprocess.py
2. python split_by_year.py
3. python models/train_arima.py
4. python models/train_xgboost.py
5. python models/train_lstm.py
6. python validation/validate_all_models.py
7. python models/shap_analysis.py
8. streamlit run app.py

Results will be saved in:
results/model_comparison.csv
