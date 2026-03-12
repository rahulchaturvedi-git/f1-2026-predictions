import fastf1
import pandas as pd
from src.utils.config import *

YEAR = 2026
RACE = "Australian Grand Prix"

session = fastf1.get_session(YEAR, RACE, "Q")
session.load()

results = session.results

df = results[
    [
        "Abbreviation",
        "GridPosition",
        "Q3"
    ]
]

df.rename(columns={"Abbreviation": "Driver"}, inplace=True)

df["Q3"] = pd.to_timedelta(df["Q3"]).dt.total_seconds()

df.to_csv("data/raw/qualifying.csv", index=False)

print("Qualifying data saved")