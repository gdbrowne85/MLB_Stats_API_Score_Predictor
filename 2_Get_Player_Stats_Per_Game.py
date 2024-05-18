import requests
import pandas as pd
import time
from datetime import datetime

# Load game IDs by date from Excel
df_game_ids = pd.read_excel('mlb_games24.xlsx')

# Setup a requests session
session = requests.Session()

# API base URL
base_url = "https://statsapi.mlb.com/api/v1/"

def extract_player_stats(players, game_id, game_date, team_type, player_data):
    for player_id, details in players.items():
        player_stats = details.get('stats', {})
        position = details.get('position', {}).get('abbreviation', 'N/A')

        player_info = {
            "Game ID": game_id,
            "Game Date": game_date,
            "Team Type": team_type,
            "Player ID": details['person']['id'],
            "Player Name": details['person']['fullName'],
            "Position": position,
            # Initialize all potential stats fields with zero
            "Outs Recorded": 0,
            "Earned Runs": 0,
            "Strike Outs": 0,
            "Walks Allowed": 0,
            "Hits Allowed": 0,
            "At Bats": 0,
            "Hits": 0,
            "Walks Earned": 0,
            "Total Bases": 0
        }

        # Check the position code to determine role and extract stats accordingly
        if position == 'P':  # Assuming 'P' stands for Pitcher
            player_info['Role'] = "Pitcher"
            if 'pitching' in player_stats:
                pitching_stats = player_stats['pitching']
                player_info.update({
                    "Outs Recorded": pitching_stats.get('outs', 0),
                    "Earned Runs": pitching_stats.get('earnedRuns', 0),
                    "Strike Outs": pitching_stats.get('strikeOuts', 0),
                    "Walks Allowed": pitching_stats.get('baseOnBalls', 0),
                    "Hits Allowed": pitching_stats.get('hits', 0)
                })

        else:  # Consider other positions as hitters
            player_info['Role'] = "Hitter"
            if 'batting' in player_stats:
                batting_stats = player_stats['batting']
                player_info.update({
                    "At Bats": batting_stats.get('atBats', 0),
                    "Hits": batting_stats.get('hits', 0),
                    "Walks Earned": batting_stats.get('baseOnBalls', 0),
                    "Total Bases": batting_stats.get('totalBases', 0),
                })

        player_data.append(player_info)



# Iterate through each game
all_player_data = []
headers = {"accept": "application/json"}
total_games = len(df_game_ids)
completed_games = 0
start_time = datetime.now()

for index, row in df_game_ids.iterrows():
    game_id = row['Game ID']
    game_date = row['Date']
    url = f"{base_url}game/{game_id}/boxscore"
    try:
        response = session.get(url, headers=headers)
        data = response.json()
        teams_data = data.get('teams', {})
        # Extract stats for both home and away teams
        if 'home' in teams_data:
            extract_player_stats(teams_data['home']['players'], game_id, game_date, 'Home', all_player_data)
        if 'away' in teams_data:
            extract_player_stats(teams_data['away']['players'], game_id, game_date, 'Away', all_player_data)
    except requests.RequestException as e:
        print(f"Request failed for game ID {game_id}: {e}")
        continue

    completed_games += 1
    elapsed_time = (datetime.now() - start_time).total_seconds()
    estimated_total_time = (elapsed_time / completed_games) * total_games
    estimated_remaining_time = estimated_total_time - elapsed_time
    print(f'Progress: {completed_games / total_games * 100:.2f}%, Estimated Time Remaining: {estimated_remaining_time / 60:.2f} minutes')

# Save the data to an Excel file
df_players = pd.DataFrame(all_player_data)
df_players.to_excel('player_stats_from_boxscore24.xlsx', index=False)
print("Excel file has been created with all player stats.")
