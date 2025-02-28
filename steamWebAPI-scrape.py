import os
import re
import requests
import json
import nltk
from langdetect import detect, DetectorFactory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.matutils import sparse2full

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set seed for language detection consistency
DetectorFactory.seed = 0

# Function to check if text is in English using langdetect
def is_english(text):
    try:
        return detect(text) == 'en'
    except Exception as e:
        # In case language detection fails, consider the text non-English
        return False

# Define custom stopwords
custom_ignore_words = [
    'game', 'http', 'ubisoft', 'quot', 'csgo', 'strong', 'read', 'new',
    'added', 'removed', 'noopener', 'nbsp', 'apos', 'valve', 'like',
    'really', 'https', 'also', 'one', 'two', 'update', 'patch', 'steam',
    'play', 'rust', 'dayz', 'pubg', 'apex', 'legends', 'team', 'fortress',
    '2', 'counter', 'strike', 'global', 'offensive', 'cs', 'go', 'rainbow',
    'six', 'siege', 'delta', 'force', 'x', 'marvel', 'rivals'
]
stop_words = set(stopwords.words('english')).union(set(custom_ignore_words))

# Top 10 FPS games and their AppIDs
top_fps_games = {
    'PUBG: BATTLEGROUNDS': 578080,
    'Marvel Rivals': 1961460,
    'Rust': 252490,
    'Apex Legends': 1172470,
    'Delta Force': 1742020,
    'Tom Clancy\'s Rainbow Six Siege': 359550,
    'DayZ': 221100,
    'Crosshair X': 654700,
    'Team Fortress 2': 440,
    'Counter-Strike 2': 730
}

# Base URL for Steam API news
url_news = "http://api.steampowered.com/ISteamNews/GetNewsForApp/v0002/"

# Function to fetch patch notes for a given AppID
def fetch_patch_notes(appid):
    params = {"appid": appid, "count": 10000, "maxlength": 30000, "format": "json"}
    response = requests.get(url_news, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'appnews' in data and 'newsitems' in data['appnews']:
            return data['appnews']['newsitems']
    return []

# Directory to save patch notes
os.makedirs('patch_notes', exist_ok=True)

# Fetch and save patch notes for each game, filtering out non-English texts
all_documents = []
for game, appid in top_fps_games.items():
    patch_notes = fetch_patch_notes(appid)
    if patch_notes:
        # Filter patch notes: only keep those whose 'contents' are detected as English
        english_patch_notes = []
        for note in patch_notes:
            if 'contents' in note and is_english(note['contents']):
                english_patch_notes.append(note)
            else:
                print(f"Skipped a non-English patch note for '{game}'.")
        
        if english_patch_notes:
            filename = f'{appid}_patch_notes.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(english_patch_notes, f, ensure_ascii=False, indent=4)
            print(f"Saved {len(english_patch_notes)} English patch notes for '{game}' to '{filename}'")
            all_documents.extend(note['contents'] for note in english_patch_notes if 'contents' in note)
        else:
            print(f"No English patch notes found for '{game}'.")
    else:
        print(f"No patch notes found for '{game}'.")
