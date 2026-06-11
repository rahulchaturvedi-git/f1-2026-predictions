import pandas as pd
import numpy as np
import os

# PATHS
dir_map = {
    "Australia": "../../australian_gp",
    "China": "../../chinese_gp",
    "Miami": "../../miami_gp"
}

for gp, path in dir_map.items():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    processed_path = os.path.join(base_dir, "data/processed/dataset.csv")
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        # Drop seasonal features if they exist (rolling back failed experiment)
        cols_to_drop = ["Season_Avg_Finish", "Season_Avg_Grid"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        # Ensure schema matches exactly
        target_cols = [
            "Driver", "Quali_time", "Quali_delta", "GridPosition",
            "Sprint_performance_score", "Race_pace_norm", "Tyre_deg_norm",
            "Pace_Stability", "Team_Pace", "Prev_Race_Score", "Racecraft", "FinalPosition", "Race"
        ]
        # Re-save
        df = df[target_cols]
        df.to_csv(processed_path, index=False)
        print(f"✅ {gp} schema cleaned and restored")
