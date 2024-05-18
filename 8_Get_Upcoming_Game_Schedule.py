import requests
import pandas as pd
from datetime import datetime

def fetch_relief_pitchers(team_id):
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster/depthChart"
    response = requests.get(url)
    relief_pitchers = []
    if response.status_code == 200:
        roster = response.json().get('roster', [])
        for player in roster:
            if player['position']['code'] == '1':  # Code '1' typically stands for pitchers
                player_id = player['person']['id']
                player_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}?hydrate=stats(group=pitching,type=season,season=2024)"
                player_response = requests.get(player_url)
                if player_response.status_code == 200:
                    player_data = player_response.json()['people'][0]
                    stats = player_data.get('stats', [])
                    if stats:
                        pitching_stats = stats[0]['splits'][0]['stat']
                        games_started = pitching_stats.get('gamesStarted', 0)
                        if games_started == 0:
                            relief_pitchers.append(player_id)
    return relief_pitchers

url = "https://statsapi.mlb.com/api/v1/schedule/?sportId=1&hydrate=lineups,probablePitcher"

# Set the start and end dates for the game range
params = {
    'startDate': '2024-05-13',
    'endDate': '2024-05-13'
}

# Perform the API request
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    games = data.get('dates', [])[0].get('games', [])  # Assuming there's data for the date
    game_list = []

    for game in games:
        game_pk = game.get('gamePk')
        game_date = datetime.strptime(game.get('gameDate'), '%Y-%m-%dT%H:%M:%SZ').date()
        home_team_id = game['teams']['home']['team']['id']
        away_team_id = game['teams']['away']['team']['id']
        home_team = game['teams']['home']['team']['name']
        away_team = game['teams']['away']['team']['name']

        # Probable pitchers
        home_pitcher_id = game['teams']['home'].get('probablePitcher', {}).get('id', 'No pitcher listed')
        away_pitcher_id = game['teams']['away'].get('probablePitcher', {}).get('id', 'No pitcher listed')

        # Lineups
        home_lineup = [player['id'] for player in game.get('lineups', {}).get('homePlayers', [])]
        away_lineup = [player['id'] for player in game.get('lineups', {}).get('awayPlayers', [])]

        # Fetch relief pitchers
        home_relief_pitchers = fetch_relief_pitchers(home_team_id)
        away_relief_pitchers = fetch_relief_pitchers(away_team_id)

        game_list.append({
            'game_date': game_date,
            'game_id': game_pk,
            'home_team': home_team,
            'away_team': away_team,
            'home_starting_pitcher': home_pitcher_id,
            'away_starting_pitcher': away_pitcher_id,
            'home_lineup': home_lineup,
            'away_lineup': away_lineup,
            'home_relief_pitchers': home_relief_pitchers,
            'away_relief_pitchers': away_relief_pitchers
        })

    # Create a DataFrame
    df = pd.DataFrame(game_list)

    # Save to an Excel file
    df.to_excel('upcoming_mlb_games.xlsx', index=False)
    print("Data has been saved to upcoming_mlb_games.xlsx.")
else:
    print(f"Failed to fetch data: {response.status_code} - {response.text}")
