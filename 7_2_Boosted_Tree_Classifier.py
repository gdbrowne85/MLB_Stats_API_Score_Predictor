import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Load the Excel files
df_2022 = pd.read_excel('classifier_features_response22.xlsx')  # Load data from 2022
df_2023 = pd.read_excel('classifier_features_response23.xlsx')  # Load data from 2023

# Concatenate the two DataFrames vertically, assuming they have the same columns
combined_df = pd.concat([df_2022, df_2023], ignore_index=True)

# Save the combined DataFrame to a new Excel file
combined_df.to_excel('combined_classifier_features_response.xlsx', index=False)  # Adjust the path and filename as necessary

print("Data combined and saved successfully.")

# Load the data
data = pd.read_excel('combined_classifier_features_response.xlsx')  # Adjust the path to where you've saved your Excel file

# Preprocessing
data.drop(columns=['Game ID', 'game_id'], inplace=True)  # Drop unwanted columns
data.dropna(inplace=True)  # Drop rows with missing values

# Define features and target
X = data.drop('home_win', axis=1)
y = data['home_win']

#Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
scaler_path = 'scaler.joblib'
dump(scaler, scaler_path)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=3)

# Set up the hyperparameter grid to tune
params = {
    'max_depth': [3],
    'n_estimators': [1000],
    'learning_rate': [0.01]
}

# Create and train the XGBoost model with GridSearchCV
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
grid_search = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=3)
grid_search.fit(X_train, y_train)

# Best model after grid search
best_model = grid_search.best_estimator_

# Predict on the validation set with the best model
y_pred = best_model.predict(X_val)
y_proba = best_model.predict_proba(X_val)  # Get probabilities for each class

# Adjust probabilities so they always reflect confidence of the predicted class
confidence_scores = y_proba[range(len(y_pred)), y_pred]

# Calculate and print the accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f'Best Model Parameters: {grid_search.best_params_}')
print(f'Validation Accuracy of the Best Model: {accuracy:.2f}')

# Create a DataFrame with actual values, predicted values, and confidence scores
results_df = pd.DataFrame({
    'Actual': y_val,
    'Predicted': y_pred,
    'Confidence Score': confidence_scores
})

# Reset index to ensure the DataFrame looks clean when saved (optional)
results_df.reset_index(drop=True, inplace=True)

# Thresholds for confidence score
thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Calculate and print accuracy for various confidence thresholds
for threshold in thresholds:
    filtered_df = results_df[results_df['Confidence Score'] > threshold]
    if not filtered_df.empty:
        filtered_accuracy = accuracy_score(filtered_df['Actual'], filtered_df['Predicted'])
        print(f'Accuracy for confidence > {threshold:.2f}: {filtered_accuracy:.2f}')
    else:
        print(f'No predictions with confidence > {threshold:.2f}')

# Save the results to an Excel file
results_path = 'output_predictions_xgboost.xlsx'  # Specify your desired output file name
results_df.to_excel(results_path, index=False)
print(f'Results saved to {results_path}')

# After training and finding the best model
best_model = grid_search.best_estimator_

# Save the model to a file
model_path = 'best_xgboost_model.json'  # Saving as JSON file (recommended for compatibility reasons)
best_model.save_model(model_path)
print(f"Model saved successfully to {model_path}")
