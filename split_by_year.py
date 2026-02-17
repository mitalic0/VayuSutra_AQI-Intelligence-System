
import pandas as pd

df = pd.read_csv("data/processed_aqi.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year

train = df[df['Year'] <= 2021]
validation = df[df['Year'] > 2021]

train.to_csv("data/train.csv", index=False)
validation.to_csv("data/validation.csv", index=False)

print("Train/Validation split completed.")
