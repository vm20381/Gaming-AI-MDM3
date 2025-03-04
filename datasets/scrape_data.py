import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

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
        return game_name, ", ".join(tags)
    else:
        print(f"Failed to fetch Steam tags for {game_id}")
        return f"Unknown ({game_id})", "N/A"
    
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
            return df
        else:
            print(f"Could not find the data table for game {game_id}.")
            return None
    else:
        print(f"Failed to fetch SteamCharts data for {game_id}: {response.status_code}")
        return None

def save_game_data(game_name):
    game_id = get_steam_app_id(game_name) 
    if not game_id:
        return
    
    game_name, tags = get_steam_tags(game_id) 
    steamcharts_data = get_steamcharts_data(game_id) 

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
    game_names = ["Counter-Strike 2","PUBG: BATTLEGROUNDS","Dota 2","NARAKA: BLADEPOINT","Marvel Rivals","Apex Legends","Delta Force","War Thunder","Crab Game","Team Fortress 2","VRChat","Once Human",
                  "Path of Exile","Yu-Gi-Oh! Master Duel","Destiny 2","Overwatch 2","Russian Fishing 4","MIR4","Eternal Return","Unturned"] # "THRONE AND LIBERTY","Lost Ark","Warframe" perms error
    for game in game_names:
        save_game_data(game)