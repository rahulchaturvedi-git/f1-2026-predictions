import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    "/home/rahul/projects/f1-2026-predictions/dataset/season_dataset.csv"
)

train_df = df[df["Race"] == "Australia"]
test_df = df[df["Race"] == "China"].copy()

drop_cols = ["Driver", "FinalPosition", "Race"]

X_train = train_df.drop(columns=drop_cols).fillna(0)
y_train = train_df["FinalPosition"]

X_test = test_df.drop(columns=drop_cols).fillna(0)

# SCALE
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SIMPLE MODEL (LESS OVERFIT)
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# PREDICT
preds = model.predict(X_test)
test_df["PredictedScore"] = preds

# PROBABILITY
exp_scores = np.exp(-preds)
test_df["WinProbability"] = exp_scores / np.sum(exp_scores)

test_df = test_df.sort_values("WinProbability", ascending=False)

print("\n🏁 WIN PROBABILITIES:\n")
print(test_df[["Driver", "GridPosition", "WinProbability"]])

print("\n🥇 TOP 3:\n")
print(test_df.head(3)[["Driver", "WinProbability"]])

# FEATURE IMPORTANCE
importances = pd.Series(
    model.feature_importances_,
    index=train_df.drop(columns=drop_cols).columns
)

print("\n🔥 FEATURE IMPORTANCE:\n")
print(importances.sort_values(ascending=False))