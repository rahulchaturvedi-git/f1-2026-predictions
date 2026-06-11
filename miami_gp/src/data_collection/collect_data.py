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

# -----------------------
# CONFIG & PATHS
# -----------------------
YEAR = 2026
RACE = 'Miami Grand Prix'
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
RAW_PATH = os.path.join(BASE_PATH, 'raw')
CACHE_PATH = os.path.join(BASE_PATH, 'cache')

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)

# Enable FastF1 Cache
fastf1.Cache.enable_cache(CACHE_PATH)

def fetch_fp1():
    """Fetches FP1 laps data."""
    print("📡 Fetching FP1...")
    session = fastf1.get_session(YEAR, RACE, 'FP1')
    session.load()
    laps = session.laps[['Driver', 'LapTime']].dropna()
    laps['LapTime'] = laps['LapTime'].dt.total_seconds()
    
    output_file = os.path.join(RAW_PATH, "fp1.csv")
    laps.to_csv(output_file, index=False)
    print(f"✅ FP1 data saved to {output_file}")

def fetch_sprint_data():
    """Fetches Sprint Qualifying and Sprint Race data."""
    # Sprint Qualifying
    print("📡 Fetching Sprint Qualifying...")
    sq = fastf1.get_session(YEAR, RACE, 'SQ')
    sq.load()
    sq_laps = sq.laps[['Driver', 'LapTime']].dropna()
    sq_laps['LapTime'] = sq_laps['LapTime'].dt.total_seconds()
    
    sq_file = os.path.join(RAW_PATH, "sprint_quali.csv")
    sq_laps.to_csv(sq_file, index=False)
    print(f"✅ Sprint Qualifying data saved to {sq_file}")

    # Sprint Race
    print("📡 Fetching Sprint Race...")
    s = fastf1.get_session(YEAR, RACE, 'S')
    s.load()
    s_results = s.results[['Abbreviation', 'Position']].copy()
    s_results.columns = ['Driver', 'SprintPosition']
    
    s_file = os.path.join(RAW_PATH, "sprint_race.csv")
    s_results.to_csv(s_file, index=False)
    print(f"✅ Sprint Race results saved to {s_file}")

    # Sprint Laps for Pace Analysis
    s_laps = s.laps[['Driver', 'LapTime']].dropna().copy()
    s_laps['LapTime'] = s_laps['LapTime'].dt.total_seconds()
    sl_file = os.path.join(RAW_PATH, "sprint_laps.csv")
    s_laps.to_csv(sl_file, index=False)
    print(f"✅ Sprint Race laps saved to {sl_file}")

def fetch_race_weekend_data():
    """Fetches Race Qualifying and Race Results."""
    # Race Qualifying
    print("📡 Fetching Race Qualifying...")
    q = fastf1.get_session(YEAR, RACE, 'Q')
    q.load()
    q_laps = q.laps[['Driver', 'LapTime']].dropna()
    q_laps['LapTime'] = q_laps['LapTime'].dt.total_seconds()
    
    q_file = os.path.join(RAW_PATH, "race_quali.csv")
    q_laps.to_csv(q_file, index=False)
    print(f"✅ Race Qualifying data saved to {q_file}")

    # Race Results
    print("📡 Fetching Race Results...")
    r = fastf1.get_session(YEAR, RACE, 'R')
    r.load()
    r_results = r.results[['Abbreviation', 'Position']].copy()
    r_results.columns = ['Driver', 'Position']
    r_results.rename(columns={'Position': 'Position'}, inplace=True) # Ensure consistency

    r_file = os.path.join(RAW_PATH, "race_results.csv")
    r_results.to_csv(r_file, index=False)
    print(f"✅ Race results saved to {r_file}")

if __name__ == "__main__":
    print(f"🏁 Starting data collection for {YEAR} {RACE} (Sprint Weekend)...")
    fetch_fp1()
    fetch_sprint_data()
    fetch_race_weekend_data()
    print("✨ All data collection completed successfully!")
