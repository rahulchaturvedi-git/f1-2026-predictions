import pandas as pd
import os

# -----------------------
# PATHS
# -----------------------
BASE_DIR = "/home/rahul/projects/f1-2026-predictions"
OUTPUT_PATH = os.path.join(BASE_DIR, "dataset", "season_dataset.csv")

# -----------------------
# FIND RACE FOLDERS
# -----------------------
race_folders = [
    folder for folder in os.listdir(BASE_DIR)
    if folder.endswith("_gp")
]

all_dfs = []

# -----------------------
# LOAD EACH RACE DATASET
# -----------------------
for race in race_folders:
    dataset_path = os.path.join(
        BASE_DIR,
        race,
        "data",
        "processed",
        "dataset.csv"
    )

    if os.path.exists(dataset_path):
        print(f"✅ Loading {race}")
        df = pd.read_csv(dataset_path)
        all_dfs.append(df)
    else:
        print(f"⚠️ Skipping {race} (no dataset found)")

# -----------------------
# MERGE
# -----------------------
if not all_dfs:
    raise ValueError("❌ No datasets found to merge")

merged_df = pd.concat(all_dfs, ignore_index=True)

# -----------------------
# SORT (IMPORTANT)
# -----------------------
# Helps later for time-based training
merged_df = merged_df.sort_values(by=["Race", "FinalPosition"])

# -----------------------
# SAVE
# -----------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
merged_df.to_csv(OUTPUT_PATH, index=False)

print("🚀 Season dataset created successfully!")
print(f"📁 Saved at: {OUTPUT_PATH}")
print(f"📊 Total rows: {len(merged_df)}")