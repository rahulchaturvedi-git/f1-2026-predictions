import pandas as pd
import joblib

# load trained model
model = joblib.load("models/f1_winner_model.pkl")

# load feature data (NO race result here)
practice = pd.read_csv("data/raw/practice_laps.csv")
quali = pd.read_csv("data/raw/qualifying.csv")

# merge features
dataset = practice.merge(quali, on="Driver")

# features used during training
X = dataset[
    [
        "FP1_avg",
        "FP2_avg",
        "FP3_avg",
        "Q3",
        "GridPosition"
    ]
]

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
predictions.to_csv("outputs/race_prediction.csv", index=False)

print("\nPredictions saved to outputs/race_prediction.csv")