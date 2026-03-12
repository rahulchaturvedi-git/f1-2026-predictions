import fastf1
import pandas as pd
from src.utils.config import *

YEAR = 2026
RACE = "Australian Grand Prix"

session = fastf1.get_session(YEAR, RACE, "R")
session.load()

results = session.results

df = results[
    [
        "Abbreviation",
        "Position"
    ]
]

df.rename(columns={"Abbreviation": "Driver"}, inplace=True)

df.to_csv("data/raw/race_results.csv", index=False)

print("Race result label saved")