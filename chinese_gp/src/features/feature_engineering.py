import pandas as pd
import numpy as np
import os

# PATHS
CHN_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CHN_RAW_PATH = os.path.join(CHN_BASE_DIR, "data/raw")
CHN_PROCESSED_PATH = os.path.join(CHN_BASE_DIR, "data/processed/dataset.csv")
CHN_PREV_GP_DATA = os.path.abspath(os.path.join(CHN_BASE_DIR, "../australian_gp/data/processed/dataset.csv"))

# DRIVER → TEAM
driver_to_team = {
    "VER": "RB", "HAD": "RB",
    "LEC": "FER", "HAM": "FER",
    "RUS": "MER", "ANT": "MER",
    "NOR": "MCL", "PIA": "MCL",
    "ALO": "AST", "STR": "AST",
    "GAS": "ALP", "COL": "ALP",
    "LIN": "RB2", "LAW": "RB2",
    "ALB": "WIL", "SAI": "WIL",
    "OCO": "HAA", "BEA": "HAA",
    "HUL": "AUD", "BOR": "AUD",
    "PER": "CAD", "BOT": "CAD",
}

# LOAD
fp1 = pd.read_csv(f"{CHN_RAW_PATH}/fp1.csv")
sprint_laps = pd.read_csv(f"{CHN_RAW_PATH}/sprint_laps.csv")
sq = pd.read_csv(f"{CHN_RAW_PATH}/sprint_quali.csv")
sprint = pd.read_csv(f"{CHN_RAW_PATH}/sprint_race.csv")
rq = pd.read_csv(f"{CHN_RAW_PATH}/race_quali.csv")
results = pd.read_csv(f"{CHN_RAW_PATH}/race_results.csv")

# CLEAN LAPS
def clean_laps(df):
    df = df[df["LapTime"].notna()].copy()
    if df["LapTime"].dtype != float:
        df["LapTime"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    df = df[df["LapTime"] < df["LapTime"].quantile(0.9)]
    return df

# PACE (FP1 + Sprint Race Laps)
fp_all = pd.concat([fp1, sprint_laps])

def compute_race_pace(df):
    df = clean_laps(df)
    pace = {}
    for d, g in df.groupby("Driver"):
        laps = np.sort(g["LapTime"].values)
        if len(laps) < 5: pace[d] = 0; continue
        stable = laps[int(0.2*len(laps)):int(0.8*len(laps))]
        pace[d] = np.mean(stable)
    return pd.Series(pace, name="Race_pace").rename_axis("Driver")

def compute_tyre_degradation(df):
    df = clean_laps(df)
    degradation = {}
    for driver, group in df.groupby("Driver"):
        group = group.sort_index()
        laps = group["LapTime"].values
        lap_nums = np.arange(len(laps))
        if len(laps) < 6: degradation[driver] = 0; continue
        slope = np.polyfit(lap_nums, laps, 1)[0]
        degradation[driver] = slope
    return pd.Series(degradation, name="Tyre_deg").rename_axis("Driver")

race_pace_df = compute_race_pace(fp_all).reset_index()
tyre_deg_df = compute_tyre_degradation(fp_all).reset_index()

# MAIN QUALI
rq_feat = rq.groupby("Driver")["LapTime"].min().reset_index()
rq_feat.rename(columns={"LapTime": "Quali_time"}, inplace=True)
pole = rq_feat["Quali_time"].min()
rq_feat["Quali_delta"] = rq_feat["Quali_time"] - pole
rq_feat = rq_feat.sort_values("Quali_time").reset_index(drop=True)
rq_feat["GridPosition"] = rq_feat.index + 1

# SPRINT
sq_rank = sq.groupby("Driver")["LapTime"].min().reset_index().sort_values("LapTime")
sq_rank["SprintGrid"] = range(1, len(sq_rank) + 1)
df = rq_feat.merge(sprint[["Driver", "SprintPosition"]], on="Driver")
df = df.merge(sq_rank[["Driver", "SprintGrid"]], on="Driver")
df["Sprint_performance_score"] = df["SprintGrid"] - df["SprintPosition"]

# MERGE
df = df.merge(race_pace_df, on="Driver", how="left")
df = df.merge(tyre_deg_df, on="Driver", how="left")
df["Pace_Stability"] = 0.0 # Default for CHN

# NORMALIZATION
max_pace = df["Race_pace"].max()
df["Race_pace_norm"] = df["Race_pace"] / max_pace if max_pace != 0 else 0
max_deg = df["Tyre_deg"].max()
df["Tyre_deg_norm"] = df["Tyre_deg"] / max_deg if max_deg != 0 else 0

# TEAM PACE
df["Team"] = df["Driver"].map(driver_to_team)
df["Team_Pace"] = df.groupby("Team")["Race_pace_norm"].transform("mean")
df.drop(columns=["Team"], inplace=True)

# HISTORY
prev_df = pd.read_csv(CHN_PREV_GP_DATA)
# Ensure history matching uses Pace_Stability if present, but here we just need FinalPosition
prev_map = prev_df.set_index("Driver").to_dict("index")
df["Prev_Race_Score"] = df["Driver"].apply(lambda d: 1 / prev_map.get(d, {}).get("FinalPosition", 20))
df["Racecraft"] = df["Driver"].apply(lambda d: prev_map.get(d, {}).get("GridPosition", 20) - prev_map.get(d, {}).get("FinalPosition", 20))

# TARGET
results = results.rename(columns={"Position": "FinalPosition"})
df = df.merge(results[["Driver", "FinalPosition"]], on="Driver")

# FINAL
df["Race"] = "China"
df.fillna(0, inplace=True)
df = df[[
    "Driver", "Quali_time", "Quali_delta", "GridPosition",
    "Sprint_performance_score", "Race_pace_norm", "Tyre_deg_norm",
    "Pace_Stability", "Team_Pace", "Prev_Race_Score", "Racecraft", "FinalPosition", "Race"
]]

os.makedirs(os.path.dirname(CHN_PROCESSED_PATH), exist_ok=True)
df.to_csv(CHN_PROCESSED_PATH, index=False)
print("✅ China dataset UPDATED (Schema)")