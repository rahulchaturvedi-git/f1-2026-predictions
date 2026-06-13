import requests
import typing
import builtins
import requests.adapters
import requests.cookies

# Workaround for fastf1/requests-cache compatibility issue
builtins.RequestsCookieJar = requests.cookies.RequestsCookieJar
builtins.HTTPAdapter = requests.adapters.HTTPAdapter
typing.RequestsCookieJar = requests.cookies.RequestsCookieJar
typing.HTTPAdapter = requests.adapters.HTTPAdapter

import fastf1
import os
import pandas as pd
import numpy as np

# -----------------------
# CONFIG & PATHS
# -----------------------
YEAR = 2026
RACE = 'Monaco Grand Prix'
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
RAW_PATH = os.path.join(BASE_PATH, 'raw')
CACHE_PATH = os.path.join(BASE_PATH, 'cache')

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)

# Enable FastF1 Cache
fastf1.Cache.enable_cache(CACHE_PATH)

def fetch_practice_sessions():
    """Fetches FP1, FP2, and FP3 data and saves individual and aggregated CSVs."""
    sessions = ["FP1", "FP2", "FP3"]
    all_practice_data = []

    for s in sessions:
        print(f"📡 Fetching {s}...")
        session = fastf1.get_session(YEAR, RACE, s)
        session.load()
        
        # Individual session data (Driver, LapTime)
        laps = session.laps[['Driver', 'LapTime']].dropna()
        laps['LapTime'] = pd.to_timedelta(laps['LapTime']).dt.total_seconds()
        
        output_file = os.path.join(RAW_PATH, f"{s.lower()}.csv")
        laps.to_csv(output_file, index=False)
        print(f"✅ {s} data saved to {output_file}")

        # For aggregated practice_laps.csv
        avg_laps = laps.groupby("Driver")["LapTime"].mean().reset_index()
        avg_laps.rename(columns={"LapTime": f"{s}_avg"}, inplace=True)
        all_practice_data.append(avg_laps)

    # Aggregated Practice Laps (used by predict_race.py)
    if all_practice_data:
        merged = all_practice_data[0]
        for df in all_practice_data[1:]:
            merged = merged.merge(df, on="Driver", how="outer")
        
        practice_laps_file = os.path.join(RAW_PATH, "practice_laps.csv")
        merged.to_csv(practice_laps_file, index=False)
        print(f"🚀 Aggregated practice data saved to {practice_laps_file}")

def fetch_qualifying_data():
    """Fetches Qualifying data and saves to qualifying.csv and race_quali.csv."""
    print("📡 Fetching Qualifying...")
    session = fastf1.get_session(YEAR, RACE, 'Q')
    session.load()

    # 1. qualifying.csv (Driver, GridPosition, LapTime)
    results = session.results[["Abbreviation", "GridPosition"]].copy()
    results.rename(columns={"Abbreviation": "Driver"}, inplace=True)

    # Use the fastest available qualifying time from Q1/Q2/Q3
    for time_col in ["Q1", "Q2", "Q3"]:
        if time_col in results.columns:
            results[time_col] = pd.to_timedelta(results[time_col]).dt.total_seconds()

    qualifying_cols = [c for c in ["Q1", "Q2", "Q3"] if c in results.columns]
    if qualifying_cols:
        results["LapTime"] = results[qualifying_cols].min(axis=1, skipna=True)
    else:
        laps = session.laps[["Driver", "LapTime"]].copy()
        laps = laps[laps["LapTime"].notna()].copy()
        laps["LapTime"] = pd.to_timedelta(laps["LapTime"]).dt.total_seconds()
        best_laps = laps.groupby("Driver")["LapTime"].min().reset_index()
        results = results.merge(best_laps, on="Driver", how="left")

    results_subset = results[["Driver", "GridPosition", "LapTime"]]
    qualifying_file = os.path.join(RAW_PATH, "qualifying.csv")
    results_subset.to_csv(qualifying_file, index=False)
    print(f"✅ Qualifying data saved to {qualifying_file}")

    # 2. race_quali.csv (Driver, LapTime - raw laps)
    quali_laps = session.laps[['Driver', 'LapTime']].dropna()
    quali_laps['LapTime'] = pd.to_timedelta(quali_laps['LapTime']).dt.total_seconds()
    
    race_quali_file = os.path.join(RAW_PATH, "race_quali.csv")
    quali_laps.to_csv(race_quali_file, index=False)
    print(f"✅ Raw qualifying laps saved to {race_quali_file}")

def fetch_race_results():
    """Fetches Race Results and saves to race_results.csv."""
    print("📡 Fetching Race Results...")
    session = fastf1.get_session(YEAR, RACE, 'R')
    session.load()

    results = session.results
    df = results[['Abbreviation', 'Position']].rename(columns={'Abbreviation': 'Driver'})
    
    race_results_file = os.path.join(RAW_PATH, "race_results.csv")
    df.to_csv(race_results_file, index=False)
    print(f"✅ Race results saved to {race_results_file}")

if __name__ == "__main__":
    print(f"🏁 Starting data collection for {YEAR} {RACE}...")
    fetch_practice_sessions()
    fetch_qualifying_data()
    fetch_race_results()
    print("✨ All data collection completed successfully!")
