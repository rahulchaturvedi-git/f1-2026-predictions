import fastf1
import os

CACHE_PATH = "data/cache"

os.makedirs(CACHE_PATH, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_PATH)