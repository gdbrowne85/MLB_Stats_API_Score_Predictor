import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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
# Dropping columns containing 'Relief Pitchers'
data = data.loc[:, ~data.columns.str.contains('Relief Pitchers')]
data.dropna(inplace=True)  # Drop rows with missing values

# Define features and target
X = data.drop('home_win', axis=1)
y = data['home_win']

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit and transform the data for normalization

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=10)  # Increased max_iter to ensure convergence
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)  # Get probabilities for each class

# Calculate and print the accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.2f}')

# Create DataFrame with actual values, predicted values, certainty scores, and maximum probabilities
results_df = pd.DataFrame({
    'Actual': y_val,
    'Predicted': y_pred,
    'Certainty Score': y_proba.max(axis=1)  # Maximum probability for the predicted class
})

# Reset index to ensure the DataFrame looks clean when saved (optional)
results_df.reset_index(drop=True, inplace=True)

# Thresholds for analysis
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

# Calculating and printing accuracy for various certainty thresholds
for threshold in thresholds:
    filtered_df = results_df[results_df['Certainty Score'] > threshold]
    if not filtered_df.empty:
        threshold_accuracy = accuracy_score(filtered_df['Actual'], filtered_df['Predicted'])
        print(f'Accuracy for predictions with certainty > {threshold:.1f}: {threshold_accuracy:.2f}')
    else:
        print(f'No predictions with certainty > {threshold:.1f}')

# Save the results to an Excel file
results_path = 'output_predictions.xlsx'  # Specify your desired output file name
results_df.to_excel(results_path, index=False)
print(f'Results saved to {results_path}')
