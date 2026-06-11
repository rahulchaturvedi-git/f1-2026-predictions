import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# PATHS
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEASON_DATA_PATH = os.path.join(BASE_DIR, "dataset/season_dataset.csv")

def evaluate_race(df, train_races, test_race):
    """
    Trains on train_races and tests on test_race.
    Returns MAE, Top 3 Accuracy, and Winner Accuracy.
    """
    train_df = df[df["Race"].isin(train_races)].copy()
    test_df = df[df["Race"] == test_race].copy()

    if test_df.empty:
        return None

    drop_cols = ["Driver", "FinalPosition", "Race"]
    
    # Preprocess
    X_train = train_df.drop(columns=drop_cols).fillna(0)
    y_train = train_df["FinalPosition"]
    X_test = test_df.drop(columns=drop_cols).fillna(0)
    y_test = test_df["FinalPosition"]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict
    preds = model.predict(X_test_scaled)
    test_df["PredictedPosition"] = preds

    # Metric: MAE
    mae = mean_absolute_error(y_test, preds)

    # Metric: Top 3 Accuracy (Intersection over Union or simple hit count)
    actual_top3 = set(test_df.sort_values("FinalPosition").head(3)["Driver"])
    pred_top3 = set(test_df.sort_values("PredictedPosition").head(3)["Driver"])
    top3_acc = len(actual_top3.intersection(pred_top3)) / 3.0

    # Metric: Winner Accuracy
    actual_winner = test_df.sort_values("FinalPosition").iloc[0]["Driver"]
    pred_winner = test_df.sort_values("PredictedPosition").iloc[0]["Driver"]
    winner_acc = 1 if actual_winner == pred_winner else 0

    return {
        "MAE": mae,
        "Top3_Acc": top3_acc,
        "Winner_Hit": winner_acc
    }

def main():
    if not os.path.exists(SEASON_DATA_PATH):
        print(f"❌ Season dataset not found at {SEASON_DATA_PATH}")
        return

    df = pd.read_csv(SEASON_DATA_PATH)
    races = ["Australia", "China", "Miami", "Japan", "Canada"]
    
    results = []

    print("\n📊 F1 PREDICTION PERFORMANCE ANALYSIS (Loss Metrics)\n")
    print(f"{'Race':<12} | {'MAE':<6} | {'Top 3 Acc':<10} | {'Winner Hit':<10}")
    print("-" * 50)

    for i, race in enumerate(races):
        # For the first race, we train on itself for some baseline (usually we omit it from real eval)
        if i == 0:
             train_races = [race]
        else:
             train_races = races[:i]

        metrics = evaluate_race(df, train_races, race)
        if metrics:
            results.append({**{"Race": race}, **metrics})
            print(f"{race:<12} | {metrics['MAE']:<6.2f} | {metrics['Top3_Acc']:<10.0%} | {'✅' if metrics['Winner_Hit'] else '❌':<10}")

    res_df = pd.DataFrame(results)
    print("\n📈 SEASON PERFORMANCE SUMMARY:")
    print(f"Overall Top 3 Accuracy: {res_df['Top3_Acc'].mean():.0%}")
    print(f"Average Position Error (MAE): {res_df['MAE'].mean():.2f}")
    print(f"Total Winner Hits: {res_df['Winner_Hit'].sum()} / {len(races)}")
    
    print("\n💡 INTERPRETATION:")
    print("- Top 3 Accuracy: Higher is better. Measures how many of our predicted podium finishers actually reached the podium.")
    print("- MAE: Lower is better. Measures the average distance between predicted and actual finishing positions.")

if __name__ == "__main__":
    main()
