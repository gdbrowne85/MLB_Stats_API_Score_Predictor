import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from joblib import load

# Load the pre-trained model
model_path = 'best_xgboost_model.json'
model = xgb.XGBClassifier()
model.load_model(model_path)

# Load the feature set from 'upcoming_game_features.xlsx'
data = pd.read_excel('upcoming_game_features.xlsx')

# Assume 'game_id' is the first column; keep 'game_id' for later use
game_ids = data['Game ID']

data_clean = data.dropna()  # Drop rows with any NaN values

features = data_clean.drop(columns=['Game ID'])
print(features)

# You should have the scaler saved from the training script; load it
scaler = load('scaler.joblib')

# Normalize the features using the same scaler used in training
features_scaled = scaler.transform(features)

# Predict using the loaded model
predictions = model.predict(features_scaled)
probabilities = model.predict_proba(features_scaled)

# Get the confidence scores for the predictions
confidence_scores = probabilities.max(axis=1)

# Create a DataFrame with game IDs, predictions, and confidence scores
results_df = pd.DataFrame({
    'Game ID': game_ids,
    'Prediction': predictions,
    'Confidence Score': confidence_scores
})

# Print the DataFrame
print(results_df)

# Optionally, save the results to an Excel file
results_df.to_excel('predictions_for_upcoming_games.xlsx', index=False)
print("Predictions saved to 'predictions_for_upcoming_games.xlsx'")
