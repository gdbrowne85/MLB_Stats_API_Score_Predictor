import pandas as pd

# Load the Excel file
df = pd.read_excel('trailing_stats24.xlsx')

# Helper functions to calculate ERA, WHIP, and OPS
def calculate_ERA(earned_runs, outs_recorded):
    innings = outs_recorded / 3
    return (earned_runs / innings) * 9 if innings != 0 else 0

def calculate_WHIP(walks, hits, outs_recorded):
    innings = outs_recorded / 3
    return (walks + hits) / innings if innings != 0 else 0

def calculate_OPS(hits, total_bases, at_bats, walks):
    obp = (hits + walks) / (at_bats + walks) if at_bats + walks != 0 else 0
    slg = total_bases / at_bats if at_bats != 0 else 0
    return obp + slg

# Calculate stats for the last 10 and 3 games for home and away teams
for team in ['Home', 'Away']:
    for stat_period in [10, 3]:
        prefix = f'{team} Starting Pitcher'
        df[f'{prefix} ERA {stat_period}'] = df.apply(lambda row: calculate_ERA(row[f'{prefix} Earned Runs {stat_period}'],
                                                                               row[f'{prefix} Outs Recorded {stat_period}']), axis=1)
        df[f'{prefix} WHIP {stat_period}'] = df.apply(lambda row: calculate_WHIP(row[f'{prefix} Walks Allowed {stat_period}'],
                                                                                 row[f'{prefix} Hits Allowed {stat_period}'],
                                                                                 row[f'{prefix} Outs Recorded {stat_period}']), axis=1)

        prefix = f'{team} Relief Pitchers'
        df[f'{prefix} ERA {stat_period}'] = df.apply(lambda row: calculate_ERA(row[f'{prefix} Earned Runs {stat_period}'],
                                                                               row[f'{prefix} Outs Recorded {stat_period}']), axis=1)
        df[f'{prefix} WHIP {stat_period}'] = df.apply(lambda row: calculate_WHIP(row[f'{prefix} Walks Allowed {stat_period}'],
                                                                                 row[f'{prefix} Hits Allowed {stat_period}'],
                                                                                 row[f'{prefix} Outs Recorded {stat_period}']), axis=1)

        prefix = f'{team} Lineup'
        df[f'{prefix} OPS {stat_period}'] = df.apply(lambda row: calculate_OPS(row[f'{prefix} Hits {stat_period}'],
                                                                               row[f'{prefix} Total Bases {stat_period}'],
                                                                               row[f'{prefix} At Bats {stat_period}'],
                                                                               row[f'{prefix} Walks {stat_period}']), axis=1)

# Save the modified DataFrame back to Excel
df.to_excel('updated_stats24.xlsx', index=False)
