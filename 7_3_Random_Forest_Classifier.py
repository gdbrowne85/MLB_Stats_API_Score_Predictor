import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data
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
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Create and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Using 100 trees in the forest
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)

# Calculate and print the accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.2f}')
