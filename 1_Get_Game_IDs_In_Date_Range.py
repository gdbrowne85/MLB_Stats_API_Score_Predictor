import requests
import pandas as pd
import openpyxl

# API base URL
base_url = "https://statsapi.mlb.com/api/"

# Endpoint for fetching schedule
endpoint = "v1/schedule"

# Construct the full URL
url = f"{base_url}{endpoint}"

# Set the start and end dates for the game range
params = {
    'sportId': 1,  # Assuming MLB; replace with the appropriate sport ID if different
    'startDate': '2024-04-11',
    'endDate': '2024-05-12'
}

# Perform the API request
response = requests.get(url, params=params)
data = response.json()
print(data)
# Check if the request was successful
if response.status_code == 200:
    game_list = []
    # Extract gamePk from each game
    games = data.get('dates', [])
    for game_date in games:
        date = game_date.get('date')
        for game in game_date.get('games', []):
            game_pk = game.get('gamePk')
            game_list.append({'Date': date, 'Game ID': game_pk})

    # Create a DataFrame
    df = pd.DataFrame(game_list)

    # Save to an Excel file
    df.to_excel('mlb_games24.xlsx', index=False)
    print("Data has been saved to mlb_games23.xlsx.")
else:
    print(f"Failed to fetch data: {response.status_code} - {response.text}")
