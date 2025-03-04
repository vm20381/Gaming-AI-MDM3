import pandas as pd

def merge():
    game_data = pd.read_csv('game_data.csv', encoding='ISO-8859-1')
    twitch_data = pd.read_csv('twitch_game_data.csv', encoding='ISO-8859-1')  # Normal encoding doesn't work

    game_data.columns = game_data.columns.str.strip()
    twitch_data.columns = twitch_data.columns.str.strip()

    # Match column names
    twitch_data.rename(columns={'Game': 'Game_Name'}, inplace=True)

    # Fix date formats
    game_data['Date'] = pd.to_datetime(game_data['Month'], format='%b-%y').dt.strftime('%m/%Y')
    twitch_data['Date'] = twitch_data['Month'].astype(str).str.zfill(2) + '/' + twitch_data['Year'].astype(str)

    # Extract first word for partial matching
    game_data['Game_Key'] = game_data['Game_Name'].str.split().str[0]
    twitch_data['Game_Key'] = twitch_data['Game_Name'].str.split().str[0]

    # Select relevant columns from Twitch dataset
    twitch_data_filtered = twitch_data[['Game_Key', 'Date', 'Hours_watched', 'Hours_streamed', 'Peak_viewers', 'Streamers']]

    # left join merge (game main dataset)
    merged_data = pd.merge(game_data, twitch_data_filtered, how='left', on=['Game_Key', 'Date'])

    # Remove rows where all Twitch-related columns are NaN 
    twitch_columns = ['Hours_watched', 'Hours_streamed', 'Peak_viewers', 'Streamers']
    merged_data = merged_data.dropna(subset=twitch_columns, how='all')

    # Drop key column
    merged_data.drop(columns=['Game_Key'], inplace=True)

    merged_data.to_csv('merged_game_twitch_data.csv', index=False)
    print("Merged dataset saved! Rows:", len(merged_data))

if __name__ == "__main__":
    merge()
