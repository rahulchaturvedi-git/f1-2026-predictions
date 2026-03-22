import fastf1
import os
import pandas as pd

# Cache setup
cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/cache'))
os.makedirs(cache_path, exist_ok=True)
fastf1.Cache.enable_cache(cache_path)

# Load session
session = fastf1.get_session(2026, 'Chinese Grand Prix', 'S')
session.load()

results = session.results

# Process data
df = results[['Abbreviation', 'Position']].copy()
df.columns = ['Driver', 'SprintPosition']

# Save file
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/sprint_race.csv'))
df.to_csv(output_path, index=False)

print("Sprint Race data saved")