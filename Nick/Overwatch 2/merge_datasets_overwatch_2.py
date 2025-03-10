import pandas as pd
import calendar

# Load the datasets
steam_df = pd.read_csv('game_data.csv')
twitch_df = pd.read_csv('twitch_game_data.csv')

# Filter rows for Overwatch 2 (case-insensitive match on game name)
ow2_steam = steam_df[steam_df['Game_Name'].str.strip().str.lower() == 'overwatch 2'].copy()
ow2_twitch = twitch_df[twitch_df['Game'].str.strip().str.lower() == 'overwatch 2'].copy()

# Remove "Last 30 Days" entry from the Steam data, since it's not a specific month
ow2_steam = ow2_steam[ow2_steam['Month'] != 'Last 30 Days'].copy()

# Convert Month (e.g., "February 2025") into numeric month and year in the Steam data
# We'll map month names to numbers using the calendar module
month_map = {name: idx for idx, name in enumerate(calendar.month_name) if name}
# Create new columns for numeric month and year
ow2_steam['Year'] = ow2_steam['Month'].apply(lambda x: int(x.split()[-1]))
ow2_steam['Month'] = ow2_steam['Month'].apply(lambda x: month_map[x.split()[0]])

# In the Twitch data, ensure Month and Year are numeric (they should be already)
ow2_twitch['Month'] = ow2_twitch['Month'].astype(int)
ow2_twitch['Year'] = ow2_twitch['Year'].astype(int)

# Drop the redundant game name column from one side to avoid duplication.
# We'll drop 'Game' from the Twitch data and use 'Game_Name' from Steam data for consistency.
ow2_twitch = ow2_twitch.drop(columns=['Game'])

# Merge the two datasets on Year and Month using an outer join to keep all months from both.
merged_ow2 = pd.merge(ow2_steam, ow2_twitch, on=['Year', 'Month'], how='outer')

# Sort the merged data by Year and Month for chronological order
merged_ow2 = merged_ow2.sort_values(['Year', 'Month']).reset_index(drop=True)

# Save the merged dataframe to CSV
merged_ow2.to_csv('merged_overwatch2_data.csv', index=False)
