import requests

STEAMSPY_URL = "https://steamspy.com/api.php?request=all"

def get_free_to_play_games():
    try:
        response = requests.get(STEAMSPY_URL, timeout=10)
        response.raise_for_status()
        games = response.json()

        free_games = []

        for app_id, game in games.items():
            if game["price"] == "0":  # Free
                name = game["name"]
                if "demo" not in name.lower() and "test" not in name.lower():  # Filter out demos/tests
                    free_games.append({"id": app_id, "name": name})
        
        return free_games

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

if __name__ == '__main__':
    free_games = get_free_to_play_games()
    print(f"Found {len(free_games)} free-to-play games!")