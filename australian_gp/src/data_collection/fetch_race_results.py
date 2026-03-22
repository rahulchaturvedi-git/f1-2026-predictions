import fastf1
import os
import pandas as pd

# Paths
cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/cache'))
os.makedirs(cache_path, exist_ok=True)
fastf1.Cache.enable_cache(cache_path)

session = fastf1.get_session(2026, 'Australian Grand Prix', 'R')
session.load()

results = session.results
df = results[['Abbreviation', 'Position']].rename(columns={'Abbreviation': 'Driver'})

output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/race_results.csv'))
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print("Race results saved")
