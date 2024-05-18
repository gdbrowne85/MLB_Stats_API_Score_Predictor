import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from skorch import NeuralNetRegressor
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
X = data.drop('total_runs', axis=1)  # Assuming 'total_runs' is the target variable for regression
y = data['total_runs']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
scaler_path = 'scaler.joblib'
dump(scaler, scaler_path)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=3)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

# Create a PyTorch dataset and dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the PyTorch model
class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers):
        super(RegressionModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Hyperparameter grid
param_grid = {
    'lr': [0.001, 0.01, 0.1],
    'max_epochs': [50, 100, 150],
    'module__hidden_dim': [32, 64, 128],
    'module__hidden_layers': [1, 2, 3],
    'batch_size': [16, 32, 64]
}

# Create the Skorch wrapper for the PyTorch model
net = NeuralNetRegressor(
    module=RegressionModel,
    module__input_dim=X_train.shape[1],
    optimizer=optim.Adam,
    criterion=nn.MSELoss,
    train_split=None,  # Disable internal validation
    verbose=0
)

# Perform hyperparameter tuning with RandomizedSearchCV
rs = RandomizedSearchCV(net, param_grid, n_iter=20, scoring='neg_mean_squared_error', cv=3, verbose=2, n_jobs=-1, random_state=3)
rs.fit(X_train_tensor, y_train_tensor)

# Print best parameters and lowest RMSE
print(f'Best: {rs.best_score_} using {rs.best_params_}')

# Evaluate the best model on the validation set
best_model = rs.best_estimator_
y_pred = best_model.predict(X_val_tensor)
mse = mean_squared_error(y_val_tensor.detach().numpy(), y_pred)
print(f'Validation Mean Squared Error of the Best Model: {mse:.2f}')

# Create a DataFrame with actual values and predicted values
results_df = pd.DataFrame({
    'Actual': y_val_tensor.detach().numpy().flatten(),
    'Predicted': y_pred.flatten()
})

# Reset index to ensure the DataFrame looks clean when saved (optional)
results_df.reset_index(drop=True, inplace=True)

# Save the results to an Excel file
results_path = 'output_predictions_pytorch_regression.xlsx'  # Specify your desired output file name
results_df.to_excel(results_path, index=False)
print(f'Results saved to {results_path}')

# Save the best model
torch.save(best_model.module_.state_dict(), 'best_pytorch_regression_model.pth')
print(f"Model saved successfully to 'best_pytorch_regression_model.pth'")
