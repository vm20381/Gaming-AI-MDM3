import os 
import requests
import json
import nltk
from langdetect import detect, DetectorFactory
from nltk.corpus import stopwords

def load_patch_notes(directory='patch_notes', appid=None, return_all=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(base_dir, directory)
    
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return []
    
    all_patch_notes = []
    for filename in os.listdir(directory):
        if filename.endswith('_patch_notes.json'):
            file_appid = filename.split('_')[0]
            if appid and file_appid != str(appid):
                continue
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                patch_notes = json.load(file)
                all_patch_notes.extend(patch_notes)
            if appid:
                return all_patch_notes
    return all_patch_notes if return_all else []

nltk.download('punkt')
nltk.download('stopwords')
DetectorFactory.seed = 0

def is_english(text):
    try:
        return detect(text) == 'en'
    except Exception as e:
        return False

custom_ignore_words = [
    'game', 'http', 'ubisoft', 'quot', 'csgo', 'strong', 'read', 'new',
    'added', 'removed', 'noopener', 'nbsp', 'apos', 'valve', 'like',
    'really', 'https', 'also', 'one', 'two', 'update', 'patch', 'steam',
    'play', 'rust', 'dayz', 'pubg', 'apex', 'legends', 'team', 'fortress',
    '2', 'counter', 'strike', 'global', 'offensive', 'cs', 'go', 'rainbow',
    'six', 'siege', 'delta', 'force', 'x', 'marvel', 'rivals'
]
stop_words = set(stopwords.words('english')).union(set(custom_ignore_words))

top_fps_games = {
    'Overwatch 2': 	2357570
}

url_news = "http://api.steampowered.com/ISteamNews/GetNewsForApp/v0002/"

def fetch_patch_notes(appid):
    params = {"appid": appid, "count": 10000, "maxlength": 30000, "format": "json"}
    response = requests.get(url_news, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'appnews' in data and 'newsitems' in data['appnews']:
            return data['appnews']['newsitems']
    return []

os.makedirs('patch_notes', exist_ok=True)

if __name__ == '__main__':
    all_documents = []
    for game, appid in top_fps_games.items():
        patch_notes = fetch_patch_notes(appid)
        if patch_notes:
            english_patch_notes = []
            for note in patch_notes:
                if 'contents' in note and is_english(note['contents']):
                    english_patch_notes.append(note)
                else:
                    print(f"Skipped a non-English patch note for '{game}'.")
            if english_patch_notes:
                filename = os.path.join('patch_notes', f'{appid}_patch_notes.json')
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(english_patch_notes, f, ensure_ascii=False, indent=4)
                print(f"Saved {len(english_patch_notes)} English patch notes for '{game}' to '{filename}'")
                all_documents.extend(note['contents'] for note in english_patch_notes if 'contents' in note)
            else:
                print(f"No English patch notes found for '{game}'.")
        else:
            print(f"No patch notes found for '{game}'.")