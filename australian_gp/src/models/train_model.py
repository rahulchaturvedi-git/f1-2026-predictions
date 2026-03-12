import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# load dataset
data = pd.read_csv("data/processed/dataset.csv")

# features
X = data[
    [
        "FP1_avg",
        "FP2_avg",
        "FP3_avg",
        "Q3",
        "GridPosition"
    ]
]

# label
y = data["winner"]

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
os.makedirs("models", exist_ok=True)

# save model
joblib.dump(model, "models/f1_winner_model.pkl")

print("Model saved to models/f1_winner_model.pkl")