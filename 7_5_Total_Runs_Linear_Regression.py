import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from joblib import dump

# Load the Excel files
df_2022 = pd.read_excel('total_runs_features_response22.xlsx')  # Load data from 2022
df_2023 = pd.read_excel('total_runs_features_response23.xlsx')  # Load data from 2023

# Concatenate the two DataFrames vertically, assuming they have the same columns
combined_df = pd.concat([df_2022, df_2023], ignore_index=True)

# Save the combined DataFrame to a new Excel file
combined_df.to_excel('combined_total_runs_features_response.xlsx', index=False)  # Adjust the path and filename as necessary

print("Data combined and saved successfully.")

# Load the data
data = pd.read_excel('combined_total_runs_features_response.xlsx')  # Adjust the path to where you've saved your Excel file

# Preprocessing
data.drop(columns=['Game ID', 'game_id'], inplace=True)  # Drop unwanted columns
data.dropna(inplace=True)  # Drop rows with missing values

# Define features and target
X = data.drop('total_runs', axis=1)
y = data['total_runs']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
scaler_path = 'scaler.joblib'
dump(scaler, scaler_path)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=5)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)

# Calculate and print the Mean Squared Error
mse = mean_squared_error(y_val, y_pred)
print(f'Validation Mean Squared Error: {mse:.2f}')

# Create a DataFrame with actual values, predicted values, and residuals
results_df = pd.DataFrame({
    'Actual': y_val,
    'Predicted': y_pred,
    'Residual': y_val - y_pred
})

# Reset index to ensure the DataFrame looks clean when saved (optional)
results_df.reset_index(drop=True, inplace=True)

# Save the results to an Excel file
results_path = 'output_predictions_linear_regression.xlsx'  # Specify your desired output file name
results_df.to_excel(results_path, index=False)
print(f'Results saved to {results_path}')

# Save the model to a file
model_path = 'total_runs_linear_regression_model.joblib'  # Specify your desired output file name
dump(model, model_path)
print(f"Model saved successfully to {model_path}")
