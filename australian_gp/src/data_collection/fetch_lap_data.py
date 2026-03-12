import fastf1
import pandas as pd
from src.utils.config import *

YEAR = 2026
RACE = "Australian Grand Prix"

sessions = ["FP1", "FP2", "FP3"]

all_data = []

for s in sessions:

    print(f"Fetching {s}")

    session = fastf1.get_session(YEAR, RACE, s)
    session.load()

    laps = session.laps

    df = laps[["Driver", "LapTime"]]

    df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()

    df = df.groupby("Driver").mean().reset_index()

    df.rename(columns={"LapTime": f"{s}_avg"}, inplace=True)

    all_data.append(df)

merged = all_data[0]

for df in all_data[1:]:
    merged = merged.merge(df, on="Driver")

merged.to_csv("data/raw/practice_laps.csv", index=False)

print("Practice session data saved")