import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import get_games as scrape

def get_steam_app_id(game_name):
    search_url = f"https://steamcommunity.com/actions/SearchApps/{game_name}"
    response = requests.get(search_url)
    
    if response.status_code == 200 and response.json():
        app_id = response.json()[0]["appid"]  # Get first search result
        return app_id
    else:
        print(f"Could not find App ID for '{game_name}'")
        return None
    
def get_steam_tags(game_id):
    url = f"https://store.steampowered.com/app/{game_id}/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        game_name_tag = soup.find("div", class_="apphub_AppName")
        game_name = game_name_tag.text.strip() if game_name_tag else f"Unknown ({game_id})"
        tags = [tag.text.strip() for tag in soup.find_all("a", class_="app_tag")]
        multiplayer = any("multiplayer" in tag.lower() for tag in tags)
        if multiplayer:
            return game_name, ", ".join(tags)
        else:
            return None, None
    else:
        return None,None
    
def get_steamcharts_data(game_id):
    url = f"https://steamcharts.com/app/{game_id}#All"
    
    # Add user-agent header to bypass 403 errors
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find the table containing the player data (first table on the page)
        table = soup.find("table", {"class": "common-table"})
        if table:
            rows = table.find_all("tr")
            
            # Extract header (Month, Avg. Players, etc.)
            headers = [th.text.strip() for th in rows[0].find_all("th")]
            
            # Extract data rows
            data = []
            for row in rows[1:]:
                cols = row.find_all("td")
                if len(cols) == len(headers):
                    data.append([col.text.strip() for col in cols])
            
            # Convert data into DataFrame
            df = pd.DataFrame(data, columns=headers)
            if "Last 30 Days" in df['Month'].values:
                df.loc[df['Month'] == "Last 30 Days", 'Month'] = "March 2025"
            return df
        else:
            print(f"Could not find the data table for game {game_id}.")
            return None
    else:
        print(f"Failed to fetch SteamCharts data for {game_id}: {response.status_code}")
        return None

def save_game_data(game_id):
    game_name, tags = get_steam_tags(game_id) 
    steamcharts_data = get_steamcharts_data(game_id) 

    if game_name is None or tags is None:
        print(f"Skipping game {game_id}, not multiplayer or failed to fetch data.")
        return

    if steamcharts_data is not None:
        # Add metadata columns
        steamcharts_data.insert(0, "Game_Id", game_id)
        steamcharts_data.insert(1, "Game_Name", game_name)
        steamcharts_data.insert(2, "Tags", tags)

        filename = "game_data.csv"
        file_exists = os.path.exists(filename)
        
        steamcharts_data.to_csv(filename, mode='a', header=not file_exists, index=False)
        
        print(f"Data saved successfully to {filename}")
    else:
        print(f"Skipping {game_name}, no SteamCharts data available.")

if __name__ == "__main__":
    games = scrape.get_free_to_play_games()
    for game in games:
        save_game_data(game['id'])