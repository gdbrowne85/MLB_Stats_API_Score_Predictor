import pandas as pd
import requests

def get_game_info(game_id):
    # Fetch the game data from the MLB stats API
    response = requests.get(f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore")
    game_data = response.json()

    # Initialize dictionary to store game info
    game_info = {
        'home_lineup': [],
        'away_lineup': [],
        'home_starting_pitcher': None,
        'away_starting_pitcher': None,
        'home_relief_pitchers': [],
        'away_relief_pitchers': [],
        'run_differential': 0,
        'total_runs': 0,
        'home_win': 2
    }
    game_info['run_differential'] = game_data['teams']['home']['teamStats']['batting']['runs'] - game_data['teams']['away']['teamStats']['batting']['runs']
    game_info['total_runs'] = game_data['teams']['home']['teamStats']['batting']['runs'] + game_data['teams']['away']['teamStats']['batting']['runs']
    if game_info['run_differential'] > 0:
        game_info['home_win'] = 1
    else:
        game_info['home_win'] = 0

    # Process teams
    for team_type in ['home', 'away']:
        players = game_data['teams'][team_type]['players']
        for player_id, player_info in players.items():
            player_id = int(player_id.replace('ID', ''))  # Clean and convert player ID

            if player_info['position']['abbreviation'] == 'P':  # Check if player is a pitcher
                games_started = int(player_info.get('seasonStats', {}).get('pitching', {}).get('gamesStarted', 0))

                if games_started == 0:  # This pitcher has not started any games
                    game_info[f'{team_type}_relief_pitchers'].append(player_id)
                elif games_started > 0:
                    if not game_info[f'{team_type}_starting_pitcher']:  # Check if no starting pitcher has been set
                        game_info[f'{team_type}_starting_pitcher'] = player_id  # Set as starting pitcher

            # Collect lineup excluding pitchers
            if 'position' in player_info and player_info['position']['abbreviation'] != 'P':
                if len(game_info[f'{team_type}_lineup']) < 9:
                    game_info[f'{team_type}_lineup'].append(player_id)

    return game_info

# Custom function to save DataFrame to Excel
def save_to_excel(df, filename):
    with pd.ExcelWriter(filename) as writer:
        df.to_excel(writer, index=False, index_label=False)

# Load the game dates and IDs
game_ids_df = pd.read_excel('mlb_games24.xlsx')

# Fetch game info for each game ID and collect the results
results = []
for _, row in game_ids_df.iterrows():
    game_id = str(row['Game ID'])  # Convert game ID to string
    game_info = get_game_info(game_id)
    game_info['game_date'] = row['Date']
    game_info['game_id'] = int(game_id.replace('ID', ''))  # Clean up any 'ID' prefix and convert to integer
    results.append(game_info)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Convert all player IDs to integers within lists
for column in ['home_lineup', 'away_lineup', 'home_relief_pitchers', 'away_relief_pitchers']:
    results_df[column] = results_df[column].apply(lambda x: [int(player) for player in x])

# Save to Excel
save_to_excel(results_df, 'mlb_game_rosters24.xlsx')

print("Game information has been successfully saved to 'mlb_game_rosters.xlsx'.")
