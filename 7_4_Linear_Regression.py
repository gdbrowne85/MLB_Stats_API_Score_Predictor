import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_excel('regression_features_response22.xlsx')  # Adjust the path to where you've saved your Excel file

# Preprocessing
data.drop(columns=['Game ID', 'game_id'], inplace=True)  # Drop unwanted columns
# Dropping columns containing 'Relief Pitchers'
data = data.loc[:, ~data.columns.str.contains('Relief Pitchers')]
data.dropna(inplace=True)  # Drop rows with missing values

# Define features and target
X = data.drop('run_differential', axis=1)
y = data['run_differential']

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit and transform the data for normalization

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)

# Calculate and print the Root Mean Squared Error (RMSE)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f'Validation RMSE: {rmse:.2f}')

# Optionally, you might want to calculate the R-squared value to determine the model's explanatory power
r_squared = model.score(X_val, y_val)
print(f'R-squared: {r_squared:.2f}')

# Create a DataFrame with actual and predicted values
results_df = pd.DataFrame({
    'Actual': y_val,
    'Predicted': y_pred
})

# Reset index to ensure the DataFrame looks clean when saved (optional)
results_df.reset_index(drop=True, inplace=True)

# Save the results to an Excel file
results_path = 'output_predictions.xlsx'  # Specify your desired output file name
results_df.to_excel(results_path, index=False)
print(f'Results saved to {results_path}')
