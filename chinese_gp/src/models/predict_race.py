import pandas as pd
import os
import pickle

# Paths
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/dataset.csv'))
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/f1_model.pkl'))
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../outputs/race_prediction.csv'))

# Load
df = pd.read_csv(data_path)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

features = [
    'FP1_avg',
    'FP1_std',
    'SQ_delta',
    'SprintPosition',
    'Sprint_gain',
    'Sprint_performance_score',
    'RQ_delta',
    'RQ_delta_norm',
    'GridPosition',
    'Quali_strength',
    'SQ_to_RQ_improvement',
    'Overall_pace_score'
]

X = df[features]

# Predict
df['PredictedPosition'] = model.predict(X)

# Rank
df = df.sort_values('PredictedPosition')

# Save
df[['Driver', 'PredictedPosition']].to_csv(output_path, index=False)

print("🏁 Race prediction:")
print(df[['Driver', 'PredictedPosition']])
print("\n✅ Saved to outputs/")