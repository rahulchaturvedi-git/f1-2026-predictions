import pandas as pd
import numpy as np
import os

# PATHS
JPN_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
JPN_RAW_PATH = os.path.join(JPN_BASE_DIR, "data/raw")
JPN_PROCESSED_PATH = os.path.join(JPN_BASE_DIR, "data/processed/dataset.csv")
JPN_PREV_GP_DATA = os.path.abspath(os.path.join(JPN_BASE_DIR, "../miami_gp/data/processed/dataset.csv"))

# DRIVER → TEAM
driver_to_team = {
    "VER": "RB", "HAD": "RB", "LEC": "FER", "HAM": "FER", "RUS": "MER", "ANT": "MER",
    "NOR": "MCL", "PIA": "MCL", "ALO": "AST", "STR": "AST", "GAS": "ALP", "COL": "ALP",
    "LIN": "RB2", "LAW": "RB2", "ALB": "WIL", "SAI": "WIL", "OCO": "HAA", "BEA": "HAA",
    "HUL": "AUD", "BOR": "AUD", "PER": "CAD", "BOT": "CAD",
}

# LOAD
fp1 = pd.read_csv(f"{JPN_RAW_PATH}/fp1.csv")
fp2 = pd.read_csv(f"{JPN_RAW_PATH}/fp2.csv")
fp3 = pd.read_csv(f"{JPN_RAW_PATH}/fp3.csv")
qualifying = pd.read_csv(f"{JPN_RAW_PATH}/qualifying.csv")
results = pd.read_csv(f"{JPN_RAW_PATH}/race_results.csv")

# CLEAN
def clean_laps(df):
    df = df[df["LapTime"].notna()].copy()
    if df["LapTime"].dtype != float:
        df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    df = df[df["LapTime"] < df["LapTime"].median() * 1.15]
    return df

# PACE
fp_all = pd.concat([fp1, fp2, fp3])
fp_all["LapNumber"] = fp_all.groupby("Driver").cumcount() + 1

def compute_stint_pace(df):
    df = clean_laps(df)
    pace = {}
    for d, g in df.groupby("Driver"):
        g = g.copy()
        g["LapDiff"] = g["LapNumber"].diff().fillna(1)
        g["StintID"] = (g["LapDiff"] > 1).cumsum()
        stints = [s for id, s in g.groupby("StintID") if len(s) >= 5]
        if not stints:
            pace[d] = g["LapTime"].min()
        else:
            all_l = pd.concat(stints)["LapTime"]
            pace[d] = all_l.mean()
    return pd.Series(pace, name="Race_pace").rename_axis("Driver")

pace_s = compute_stint_pace(fp_all)

# DEG
def compute_deg(df):
    df = clean_laps(df)
    deg = {}
    for d, g in df.groupby("Driver"):
        g = g.copy()
        g["LapDiff"] = g["LapNumber"].diff().fillna(1)
        g["StintID"] = (g["LapDiff"] > 1).cumsum()
        stints = [s for id, s in g.groupby("StintID") if len(s) >= 5]
        slopes = [np.polyfit(np.arange(len(s)), s["LapTime"].values, 1)[0] for s in stints if len(s) >= 3]
        deg[d] = np.mean(slopes) if slopes else 0
    return pd.Series(deg, name="Tyre_deg").rename_axis("Driver")

deg_s = compute_deg(fp_all)

# QUALI
rq_feat = qualifying.groupby("Driver")["LapTime"].min().reset_index().sort_values("LapTime")
rq_feat["Quali_delta"] = rq_feat["LapTime"] - rq_feat["LapTime"].min()
rq_feat["GridPosition"] = range(1, len(rq_feat) + 1)
rq_feat.rename(columns={"LapTime": "Quali_time"}, inplace=True)

# MERGE FEATURES
df = rq_feat.merge(pace_s.reset_index(), on="Driver", how="left")
df = df.merge(deg_s.reset_index(), on="Driver", how="left")
df["Sprint_performance_score"] = 0.0

# RANK NORMALIZATION
df["Race_pace_norm"] = df["Race_pace"].rank(ascending=True) / len(df)
df["Tyre_deg_norm"] = df["Tyre_deg"].rank(ascending=True) / len(df)

# TEAM PACE
df["Team"] = df["Driver"].map(driver_to_team)
df["Team_Pace"] = df.groupby("Team")["Race_pace_norm"].transform("mean")
df.drop(columns=["Team"], inplace=True)

# HISTORY
if os.path.exists(JPN_PREV_GP_DATA):
    prev_df = pd.read_csv(JPN_PREV_GP_DATA)
    prev_map = prev_df.set_index("Driver").to_dict("index")
    df["Prev_Race_Score"] = df["Driver"].apply(lambda d: 1 / prev_map.get(d, {}).get("FinalPosition", 20))
    df["Racecraft"] = df["Driver"].apply(lambda d: prev_map.get(d, {}).get("GridPosition", 20) - prev_map.get(d, {}).get("FinalPosition", 20))
else:
    df["Prev_Race_Score"] = 0.05; df["Racecraft"] = 0

# TARGET
results = results.rename(columns={"Position": "FinalPosition"})
df = df.merge(results[["Driver", "FinalPosition"]], on="Driver")

# FINAL
df["Race"] = "Japan"
df.fillna(0, inplace=True)
df = df[[
    "Driver", "Quali_time", "Quali_delta", "GridPosition",
    "Sprint_performance_score", "Race_pace_norm", "Tyre_deg_norm",
    "Team_Pace", "Prev_Race_Score", "Racecraft", "FinalPosition", "Race"
]]

os.makedirs(os.path.dirname(JPN_PROCESSED_PATH), exist_ok=True)
df.to_csv(JPN_PROCESSED_PATH, index=False)
print("✅ Japan dataset RANKED (Stability Fixed)")