import pandas as pd
from datetime import datetime, timedelta

# Load data
boxscore_data = pd.read_excel('player_stats_from_boxscore23.xlsx')
roster_data = pd.read_excel('mlb_game_rosters23.xlsx')

# Convert relevant columns to string/object type
roster_data['home_starting_pitcher'] = roster_data['home_starting_pitcher'].astype(str)
roster_data['away_starting_pitcher'] = roster_data['away_starting_pitcher'].astype(str)

# Ensure 'Player ID' and dates are in the correct format
boxscore_data['Game Date'] = pd.to_datetime(boxscore_data['Game Date'])
roster_data['game_date'] = pd.to_datetime(roster_data['game_date'])
boxscore_data['Player ID'] = boxscore_data['Player ID'].astype(int)  # Ensure Player ID is integer

def parse_player_list(player_list_str):
    if isinstance(player_list_str, str):
        player_list_str = player_list_str.replace("'", "").replace('[', '').replace(']', '').strip()
        player_ids = [int(player_id) for player_id in player_list_str.split(",") if player_id]
        return player_ids
    return []

def calculate_stats_for_days(player_ids, start_date, end_date):
    relevant_stats = boxscore_data[
        (boxscore_data['Game Date'] < end_date) &
        (boxscore_data['Game Date'] >= start_date) &
        (boxscore_data['Player ID'].isin(player_ids))
    ]
    if not relevant_stats.empty:
        summed_stats = relevant_stats.agg({
            'At Bats': 'sum',
            'Hits': 'sum',
            'Walks Earned': 'sum',
            'Total Bases': 'sum',
            'Outs Recorded': 'sum',
            'Earned Runs': 'sum',
            'Strike Outs': 'sum',
            'Walks Allowed': 'sum',
            'Hits Allowed': 'sum'
        }).to_frame().T  # Convert to a DataFrame for consistent structure
        return summed_stats
    else:
        return pd.DataFrame({
            'At Bats': [0],
            'Hits': [0],
            'Walks Earned': [0],
            'Total Bases': [0],
            'Outs Recorded': [0],
            'Earned Runs': [0],
            'Strike Outs': [0],
            'Walks Allowed': [0],
            'Hits Allowed': [0]
        })

# Process games
filtered_games = roster_data[(roster_data['game_date'] >= '2023-05-10') & (roster_data['game_date'] <= '2023-08-25')]
final_data = []

for index, game in filtered_games.iterrows():
    start_date_10_days = game['game_date'] - timedelta(days=30)
    end_date = game['game_date']
    start_date_3_days = game['game_date'] - timedelta(days=7)

    home_starting_pitcher_ids = parse_player_list(game['home_starting_pitcher'])
    away_starting_pitcher_ids = parse_player_list(game['away_starting_pitcher'])

    home_lineup_ids = parse_player_list(game['home_lineup'])
    away_lineup_ids = parse_player_list(game['away_lineup'])

    home_relief_pitcher_ids = parse_player_list(game['home_relief_pitchers'])
    away_relief_pitcher_ids = parse_player_list(game['away_relief_pitchers'])

    stats_10_days_home_starting_pitcher = calculate_stats_for_days(home_starting_pitcher_ids, start_date_10_days, end_date)
    stats_3_days_home_starting_pitcher = calculate_stats_for_days(home_starting_pitcher_ids, start_date_3_days, end_date)

    stats_10_days_away_starting_pitcher = calculate_stats_for_days(away_starting_pitcher_ids, start_date_10_days, end_date)
    stats_3_days_away_starting_pitcher = calculate_stats_for_days(away_starting_pitcher_ids, start_date_3_days, end_date)

    stats_10_days_home_lineup = calculate_stats_for_days(home_lineup_ids, start_date_10_days, end_date)
    stats_3_days_home_lineup = calculate_stats_for_days(home_lineup_ids, start_date_3_days, end_date)

    stats_10_days_away_lineup = calculate_stats_for_days(away_lineup_ids, start_date_10_days, end_date)
    stats_3_days_away_lineup = calculate_stats_for_days(away_lineup_ids, start_date_3_days, end_date)

    stats_10_days_home_relief_pitcher = calculate_stats_for_days(home_relief_pitcher_ids, start_date_10_days, end_date)
    stats_3_days_home_relief_pitcher = calculate_stats_for_days(home_relief_pitcher_ids, start_date_3_days, end_date)

    stats_10_days_away_relief_pitcher = calculate_stats_for_days(away_relief_pitcher_ids, start_date_10_days, end_date)
    stats_3_days_away_relief_pitcher = calculate_stats_for_days(away_relief_pitcher_ids, start_date_3_days, end_date)

    game_data = {
        'Game ID': game['game_id'],
        'Game Date': game['game_date'],
        'Home Starting Pitcher Outs Recorded 10': stats_10_days_home_starting_pitcher['Outs Recorded'].iloc[
            0] if stats_10_days_home_starting_pitcher is not None else None,
        'Home Starting Pitcher Walks Allowed 10': stats_10_days_home_starting_pitcher['Walks Allowed'].iloc[
            0] if stats_10_days_home_starting_pitcher is not None else None,
        'Home Starting Pitcher Hits Allowed 10': stats_10_days_home_starting_pitcher['Hits Allowed'].iloc[
            0] if stats_10_days_home_starting_pitcher is not None else None,
        'Home Starting Pitcher Earned Runs 10': stats_10_days_home_starting_pitcher['Earned Runs'].iloc[
            0] if stats_10_days_home_starting_pitcher is not None else None,
        'Home Relief Pitchers Outs Recorded 10': stats_10_days_home_relief_pitcher['Outs Recorded'].iloc[
            0] if stats_10_days_home_relief_pitcher is not None else None,
        'Home Relief Pitchers Hits Allowed 10': stats_10_days_home_relief_pitcher['Hits Allowed'].iloc[
            0] if stats_10_days_home_relief_pitcher is not None else None,
        'Home Relief Pitchers Walks Allowed 10': stats_10_days_home_relief_pitcher['Walks Allowed'].iloc[
            0] if stats_10_days_home_relief_pitcher is not None else None,
        'Home Relief Pitchers Earned Runs 10': stats_10_days_home_relief_pitcher['Earned Runs'].iloc[
            0] if stats_10_days_home_relief_pitcher is not None else None,
        'Home Lineup At Bats 10': stats_10_days_home_lineup[
            'At Bats'].sum() if stats_10_days_home_lineup is not None else None,
        'Home Lineup Hits 10': stats_10_days_home_lineup[
            'Hits'].sum() if stats_10_days_home_lineup is not None else None,
        'Home Lineup Walks 10': stats_10_days_home_lineup[
            'Walks Earned'].sum() if stats_10_days_home_lineup is not None else None,
        'Home Lineup Total Bases 10': stats_10_days_home_lineup[
            'Total Bases'].sum() if stats_10_days_home_lineup is not None else None,
        'Away Starting Pitcher Outs Recorded 10': stats_10_days_away_starting_pitcher['Outs Recorded'].iloc[
            0] if stats_10_days_away_starting_pitcher is not None else None,
        'Away Starting Pitcher Walks Allowed 10': stats_10_days_away_starting_pitcher['Walks Allowed'].iloc[
            0] if stats_10_days_away_starting_pitcher is not None else None,
        'Away Starting Pitcher Hits Allowed 10': stats_10_days_away_starting_pitcher['Hits Allowed'].iloc[
            0] if stats_10_days_away_starting_pitcher is not None else None,
        'Away Starting Pitcher Earned Runs 10': stats_10_days_away_starting_pitcher['Earned Runs'].iloc[
            0] if stats_10_days_away_starting_pitcher is not None else None,
        'Away Relief Pitchers Outs Recorded 10': stats_10_days_away_relief_pitcher['Outs Recorded'].iloc[
            0] if stats_10_days_away_relief_pitcher is not None else None,
        'Away Relief Pitchers Hits Allowed 10': stats_10_days_away_relief_pitcher['Hits Allowed'].iloc[
            0] if stats_10_days_away_relief_pitcher is not None else None,
        'Away Relief Pitchers Walks Allowed 10': stats_10_days_away_relief_pitcher['Walks Allowed'].iloc[
            0] if stats_10_days_away_relief_pitcher is not None else None,
        'Away Relief Pitchers Earned Runs 10': stats_10_days_away_relief_pitcher['Earned Runs'].iloc[
            0] if stats_10_days_away_relief_pitcher is not None else None,
        'Away Lineup At Bats 10': stats_10_days_away_lineup[
            'At Bats'].sum() if stats_10_days_away_lineup is not None else None,
        'Away Lineup Hits 10': stats_10_days_away_lineup[
            'Hits'].sum() if stats_10_days_away_lineup is not None else None,
        'Away Lineup Walks 10': stats_10_days_away_lineup[
            'Walks Earned'].sum() if stats_10_days_away_lineup is not None else None,
        'Away Lineup Total Bases 10': stats_10_days_away_lineup[
            'Total Bases'].sum() if stats_10_days_away_lineup is not None else None,
        'Home Starting Pitcher Outs Recorded 3': stats_3_days_home_starting_pitcher['Outs Recorded'].iloc[
            0] if stats_3_days_home_starting_pitcher is not None else None,
        'Home Starting Pitcher Walks Allowed 3': stats_3_days_home_starting_pitcher['Walks Allowed'].iloc[
            0] if stats_3_days_home_starting_pitcher is not None else None,
        'Home Starting Pitcher Hits Allowed 3': stats_3_days_home_starting_pitcher['Hits Allowed'].iloc[
            0] if stats_3_days_home_starting_pitcher is not None else None,
        'Home Starting Pitcher Earned Runs 3': stats_3_days_home_starting_pitcher['Earned Runs'].iloc[
            0] if stats_3_days_home_starting_pitcher is not None else None,
        'Home Relief Pitchers Outs Recorded 3': stats_3_days_home_relief_pitcher['Outs Recorded'].iloc[
            0] if stats_3_days_home_relief_pitcher is not None else None,
        'Home Relief Pitchers Hits Allowed 3': stats_3_days_home_relief_pitcher['Hits Allowed'].iloc[
            0] if stats_3_days_home_relief_pitcher is not None else None,
        'Home Relief Pitchers Walks Allowed 3': stats_3_days_home_relief_pitcher['Walks Allowed'].iloc[
            0] if stats_3_days_home_relief_pitcher is not None else None,
        'Home Relief Pitchers Earned Runs 3': stats_3_days_home_relief_pitcher['Earned Runs'].iloc[
            0] if stats_3_days_home_relief_pitcher is not None else None,
        'Home Lineup At Bats 3': stats_3_days_home_lineup[
            'At Bats'].sum() if stats_3_days_home_lineup is not None else None,
        'Home Lineup Hits 3': stats_3_days_home_lineup['Hits'].sum() if stats_3_days_home_lineup is not None else None,
        'Home Lineup Walks 3': stats_3_days_home_lineup[
            'Walks Earned'].sum() if stats_3_days_home_lineup is not None else None,
        'Home Lineup Total Bases 3': stats_3_days_home_lineup[
            'Total Bases'].sum() if stats_3_days_home_lineup is not None else None,
        'Away Starting Pitcher Outs Recorded 3': stats_3_days_away_starting_pitcher['Outs Recorded'].iloc[
            0] if stats_3_days_away_starting_pitcher is not None else None,
        'Away Starting Pitcher Walks Allowed 3': stats_3_days_away_starting_pitcher['Walks Allowed'].iloc[
            0] if stats_3_days_away_starting_pitcher is not None else None,
        'Away Starting Pitcher Hits Allowed 3': stats_3_days_away_starting_pitcher['Hits Allowed'].iloc[
            0] if stats_3_days_away_starting_pitcher is not None else None,
        'Away Starting Pitcher Earned Runs 3': stats_3_days_away_starting_pitcher['Earned Runs'].iloc[
            0] if stats_3_days_away_starting_pitcher is not None else None,
        'Away Relief Pitchers Outs Recorded 3': stats_3_days_away_relief_pitcher['Outs Recorded'].iloc[
            0] if stats_3_days_away_relief_pitcher is not None else None,
        'Away Relief Pitchers Hits Allowed 3': stats_3_days_away_relief_pitcher['Hits Allowed'].iloc[
            0] if stats_3_days_away_relief_pitcher is not None else None,
        'Away Relief Pitchers Walks Allowed 3': stats_3_days_away_relief_pitcher['Walks Allowed'].iloc[
            0] if stats_3_days_away_relief_pitcher is not None else None,
        'Away Relief Pitchers Earned Runs 3': stats_3_days_away_relief_pitcher['Earned Runs'].iloc[
            0] if stats_3_days_away_relief_pitcher is not None else None,
        'Away Lineup At Bats 3': stats_3_days_away_lineup[
            'At Bats'].sum() if stats_3_days_away_lineup is not None else None,
        'Away Lineup Hits 3': stats_3_days_away_lineup['Hits'].sum() if stats_3_days_away_lineup is not None else None,
        'Away Lineup Walks 3': stats_3_days_away_lineup[
            'Walks Earned'].sum() if stats_3_days_away_lineup is not None else None,
        'Away Lineup Total Bases 3': stats_3_days_away_lineup[
            'Total Bases'].sum() if stats_3_days_away_lineup is not None else None
    }
    final_data.append(game_data)

# Create DataFrame from the collected data
final_data_df = pd.DataFrame(final_data)

# Saving results to a single sheet
with pd.ExcelWriter('trailing_stats23.xlsx') as writer:
    final_data_df.to_excel(writer, sheet_name='trailing_stats', index=False)

print("The statistics have been calculated and saved to 'trailing_stats.xlsx'.")
