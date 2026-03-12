import pandas as pd

# load data
practice = pd.read_csv("data/raw/practice_laps.csv")
quali = pd.read_csv("data/raw/qualifying.csv")
results = pd.read_csv("data/raw/race_results.csv")

# merge practice + qualifying
dataset = practice.merge(quali, on="Driver")

# merge race result
dataset = dataset.merge(results, on="Driver")

# create winner label
dataset["winner"] = dataset["Position"].apply(lambda x: 1 if x == 1 else 0)

# select useful columns
dataset = dataset[
    [
        "Driver",
        "FP1_avg",
        "FP2_avg",
        "FP3_avg",
        "Q3",
        "GridPosition",
        "winner"
    ]
]

dataset.to_csv("data/processed/dataset.csv", index=False)

print("Dataset created successfully")