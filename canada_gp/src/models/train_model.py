import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# PATHS
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SEASON_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../dataset/season_dataset.csv"))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "f1_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Load data
df = pd.read_csv(SEASON_DATA_PATH)

# Train on all previous races
train_df = df[df["Race"].isin(["Australia", "China", "Miami", "Japan"])]
# Test on current race
test_df = df[df["Race"] == "Canada"].copy()

if test_df.empty:
    raise ValueError("❌ No data found for Canada in season_dataset.csv. Run merge_dataset.py first.")

drop_cols = ["Driver", "FinalPosition", "Race"]

X_train = train_df.drop(columns=drop_cols).fillna(0)
y_train = train_df["FinalPosition"]

X_test = test_df.drop(columns=drop_cols).fillna(0)

# Save features for prediction script consistency
features = X_train.columns.tolist()

# SCALE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SIMPLE MODEL
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# PREDICT
preds = model.predict(X_test_scaled)
test_df["PredictedScore"] = preds

# PROBABILITY
exp_scores = np.exp(-preds)
test_df["WinProbability"] = exp_scores / np.sum(exp_scores)

test_df = test_df.sort_values("WinProbability", ascending=False)

print("\n🏁 CANADA WIN PROBABILITIES:\n")
print(test_df[["Driver", "GridPosition", "WinProbability"]])

print("\n🥇 TOP 3:\n")
print(test_df.head(3)[["Driver", "WinProbability"]])

# SAVE MODEL AND SCALER
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(features, os.path.join(MODEL_DIR, "feature_names.pkl"))

print(f"\n✅ Model and scaler saved to {MODEL_DIR}")

# FEATURE IMPORTANCE
importances = pd.Series(
    model.feature_importances_,
    index=features
)

print("\n🔥 FEATURE IMPORTANCE:\n")
print(importances.sort_values(ascending=False))