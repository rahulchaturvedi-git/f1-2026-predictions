import fastf1
import os

# Create correct cache path
cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/cache'))
os.makedirs(cache_path, exist_ok=True)

fastf1.Cache.enable_cache(cache_path)

session = fastf1.get_session(2026, 'Chinese Grand Prix', 'FP1')
session.load()

laps = session.laps

df = laps[['Driver', 'LapTime']].dropna()
df['LapTime'] = df['LapTime'].dt.total_seconds()

output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/fp1.csv'))
df.to_csv(output_path, index=False)

print("FP1 data saved")