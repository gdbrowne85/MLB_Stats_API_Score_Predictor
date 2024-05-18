import pandas as pd

# Load the Excel files
df_features = pd.read_excel('updated_stats24.xlsx')  # Adjust the path to your file

# Select the first column, last 20 columns from df_features and specific columns from df_response
df_features= pd.concat([df_features.iloc[:, 0], df_features.iloc[:, -20:]], axis=1)



# Save the merged DataFrame to an Excel file
df_features.to_excel('upcoming_game_features.xlsx', index=False)  # Adjust the path and filename as necessary
