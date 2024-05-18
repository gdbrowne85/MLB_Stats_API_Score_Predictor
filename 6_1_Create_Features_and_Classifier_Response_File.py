import pandas as pd

# Load the Excel files
df_features = pd.read_excel('updated_stats23.xlsx')  # Adjust the path to your file
df_response = pd.read_excel('mlb_game_rosters23.xlsx')  # Adjust the path to your MLB game rosters file

# Select the first column, last 20 columns from df_features and specific columns from df_response
df_features_selected = pd.concat([df_features.iloc[:, 0], df_features.iloc[:, -20:]], axis=1)
df_response_selected = df_response[['game_id', 'home_win']]

# Merge the DataFrames on the game ID columns
# Assuming 'Game ID' in df_features corresponds to 'game_id' in df_response and they are of the same type
df_features_response = pd.merge(df_features_selected, df_response_selected, left_on='Game ID', right_on='game_id')

# Save the merged DataFrame to an Excel file
df_features_response.to_excel('classifier_features_response23.xlsx', index=False)  # Adjust the path and filename as necessary
