import pandas as pd
import joblib
import os

# PATHS
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(BASE_DIR, "models/f1_winner_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data/processed/dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "race_prediction.csv")

# load trained model
model = joblib.load(MODEL_PATH)

# load feature data
dataset = pd.read_csv(DATA_PATH)

# features used during training
features = [
    "Quali_delta",
    "GridPosition",
    "Race_pace_norm",
    "Tyre_deg_norm",
    "Team_Pace"
]
X = dataset[features]

# predict win probability
dataset["win_probability"] = model.predict_proba(X)[:, 1]

# sort by probability
predictions = dataset.sort_values(
    "win_probability",
    ascending=False
)

print("\nPredicted Win Probabilities:\n")
print(predictions[["Driver", "win_probability"]])

# predicted winner
winner = predictions.iloc[0]["Driver"]

print("\nPredicted Winner:")
print(winner)

# save results
os.makedirs(OUTPUT_DIR, exist_ok=True)
predictions.to_csv(OUTPUT_PATH, index=False)

print(f"\nPredictions saved to {OUTPUT_PATH}")