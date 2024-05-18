import pandas as pd
import xgboost as xgb
from joblib import load

# Load the pre-trained model
model_path = 'best_xgboost_regression_model.json'
best_model = xgb.XGBRegressor()
best_model.load_model(model_path)

# Load the feature set from 'upcoming_game_features.xlsx'
data = pd.read_excel('upcoming_game_features.xlsx')

# Assume 'Game ID' is the first column; keep 'Game ID' for later use
game_ids = data['Game ID']

# Drop rows with any NaN values
data_clean = data.dropna()

# Define the features (excluding 'Game ID')
features = data_clean.drop(columns=['Game ID'])

# Load the scaler
scaler_path = 'scaler.joblib'
scaler = load(scaler_path)

# Normalize the features using the same scaler used in training
features_scaled = scaler.transform(features)

# Predict using the loaded model
predictions = best_model.predict(features_scaled)

# Create a DataFrame with game IDs and predictions
results_df = pd.DataFrame({
    'Game ID': game_ids,
    'Predicted Total Runs': predictions
})

# Print the DataFrame
print(results_df)

# Save the results to an Excel file
results_path = 'predictions_for_upcoming_games.xlsx'
results_df.to_excel(results_path, index=False)
print(f"Predictions saved to '{results_path}'")
