import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# PATHS
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "f1_winner_model.pkl")

# load dataset
data = pd.read_csv(DATA_PATH)

# features
features = [
    "Quali_delta",
    "GridPosition",
    "Race_pace_norm",
    "Tyre_deg_norm",
    "Team_Pace"
]
X = data[features]

# label (binary winner: 1 if FinalPosition is 1, else 0)
y = (data["FinalPosition"] == 1).astype(int)

# model (handle class imbalance)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    class_weight="balanced",
    random_state=42
)

# train on full dataset
model.fit(X, y)

print("Model trained successfully")

# create models folder if missing
os.makedirs(MODEL_DIR, exist_ok=True)

# save model
joblib.dump(model, MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")