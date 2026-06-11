import pandas as pd
import os
import joblib

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "f1_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "race_prediction.csv")

# Load data
df = pd.read_csv(DATA_PATH)

# Load model, scaler, and features
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)

# Prepare features
X = df[features].fillna(0)
X_scaled = scaler.transform(X)

# Predict
df['PredictedPosition'] = model.predict(X_scaled)

# Rank
df = df.sort_values('PredictedPosition')

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
df[['Driver', 'PredictedPosition']].to_csv(OUTPUT_PATH, index=False)

print("🏁 Race prediction:")
print(df[['Driver', 'PredictedPosition']])
print(f"\n✅ Saved to {OUTPUT_PATH}")