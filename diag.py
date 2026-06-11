import requests
import typing
import builtins
import requests.adapters
import requests.cookies

builtins.RequestsCookieJar = requests.cookies.RequestsCookieJar
builtins.HTTPAdapter = requests.adapters.HTTPAdapter
typing.RequestsCookieJar = requests.cookies.RequestsCookieJar
typing.HTTPAdapter = requests.adapters.HTTPAdapter

import fastf1
import logging

logging.basicConfig(level=logging.DEBUG)

def check_conn():
    urls = [
        "https://livetiming.formula1.com/static/Index.json",
        "https://ergast.com/api/f1/2023.json"
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=10)
            print(f"URL: {url} - Status: {r.status_code}")
        except Exception as e:
            print(f"URL: {url} - Error: {e}")

if __name__ == "__main__":
    print("Checking direct requests...")
    check_conn()
    print("\nChecking FastF1 schedule loading...")
    try:
        schedule = fastf1.get_event_schedule(2026)
        print("Successfully loaded 2026 schedule")
    except Exception as e:
        print(f"Failed to load 2026 schedule: {e}")
