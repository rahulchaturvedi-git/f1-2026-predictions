import pandas as pd
import os

df = pd.read_csv("dataset/season_dataset.csv")

def analyze_race(race_name, actual_top_drivers, predicted_top_drivers):
    print(f"\n--- {race_name} GP ANALYSIS ---")
    race_df = df[df["Race"] == race_name].copy()
    drivers = list(set(actual_top_drivers + predicted_top_drivers))
    subset = race_df[race_df["Driver"].isin(drivers)][
        ["Driver", "GridPosition", "Race_pace_norm", "Tyre_deg_norm", "Quali_delta", "FinalPosition"]
    ].sort_values("FinalPosition")
    print(subset.to_string(index=False))

analyze_race("Japan", ["ANT", "PIA", "LEC"], ["ANT", "RUS", "GAS"])
analyze_race("Canada", ["ANT", "HAM", "VER"], ["RUS", "ANT", "PIA"])
