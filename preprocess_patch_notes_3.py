import os
import re
import json
import pandas as pd
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

RAW_DIR = r"C:\Users\Orlan\Documents\MDM3\Gaming-AI\Gaming-AI-MDM3\datasets"
PREPROCESSED_DIR = r"C:\Users\Orlan\Documents\MDM3\Gaming-AI\Gaming-AI-MDM3\datasets"

CUSTOM_IGNORE_WORDS = [
    'steamdb', 'store', 'storeparser', 'storeparsercom', 'storeparsercomsteamdb',
    'patchnotes', 'tom', 'clancy', 'game', 'http', 'ubisoft', 'quot', 'csgo',
    'strong', 'read', 'new', 'added', 'removed', 'noopener', 'nbsp', 'apos',
    'valve', 'like', 'really', 'https', 'also', 'one', 'two', 'update', 'patch',
    'steam', 'play', 'rust', 'dayz', 'pubg', 'apex', 'legends', 'team',
    'fortress', '2', 'counter', 'strike', 'global', 'offensive', 'cs', 'go',
    'rainbow', 'six', 'siege', 'delta', 'force', 'x', 'marvel', 'rivals',
    'released', 'encouraged', 'steam_clan_image', 'battlegrounds', 'battleground', 
    'chicken', 'dinner', 'playerunknown', 'player', 'unknown', 'battle', 'royale', 
    'pubg', 'valorant', 'changes', 'games', 'gaming', 'witcher', 'sims', 'officer', 'mana', 'skyrim'
]

def load_preprocessed_notes():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, PREPROCESSED_DIR, "combined_preprocessed_patch_notes.json")
    with open(filepath, 'r', encoding='utf-8') as infile:
        notes = json.load(infile)
    return pd.DataFrame(notes, columns=['content', 'gid', 'date', 'appid'])

def preprocess_doc(doc):
    text = doc.get('contents', '')
    
    # If a game name exists, generate its n-grams and update the ignore list.
    game_name = doc.get('game')
    if game_name:
        words = game_name.split()
        ngrams = []
        for n in range(1, len(words) + 1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n]).lower()
                ngrams.append(ngram)
        # Add these n-grams to the custom ignore words if not already present.
        for ngram in ngrams:
            if ngram not in CUSTOM_IGNORE_WORDS:
                CUSTOM_IGNORE_WORDS.append(ngram)
        # Remove all occurrences of these n-grams from the text.
        for ngram in ngrams:
            pattern = r'\b' + re.escape(ngram) + r'\b'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove URLs.
    text = ' '.join(word for word in text.split() if 'http' not in word)
    # Remove common HTML tokens.
    text = text.replace("&nbsp;", " ").replace("&apos;", "'").replace("&quot;", '"')
    
    tokens = word_tokenize(text)
    # Lowercase tokens if they match any ignore word.
    tokens = [word.lower() if word.lower() in CUSTOM_IGNORE_WORDS else word for word in tokens]
    # Remove tokens that are in the ignore list.
    tokens = [word for word in tokens if word not in CUSTOM_IGNORE_WORDS]
    # Keep only alphabetic tokens or punctuation (. , ! ?).
    tokens = [word for word in tokens if word.isalpha() or word in ['.', ',', '!', '?']]
    
    text = ' '.join(tokens)
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    return text

if __name__ == "__main__":
    # Load the combined patch notes from the input file.
    input_path = os.path.join(RAW_DIR, "all_patch_notes.json")
    with open(input_path, 'r', encoding='utf-8') as infile:
        patch_notes = json.load(infile)
    
    preprocessed_notes = []
    for note in patch_notes:
        preprocessed_text = preprocess_doc(note)
        if preprocessed_text.strip():
            preprocessed_notes.append({
                'content': preprocessed_text,
                'gid': note.get('gid'),
                'date': note.get('date'),
                'appid': note.get('appid')
            })
    
    # Remove duplicate notes based on (content, gid, date).
    unique_notes = []
    seen = set()
    for note in preprocessed_notes:
        identifier = (note['content'], note['gid'], note.get('date'))
        if identifier not in seen:
            seen.add(identifier)
            unique_notes.append(note)
    
    # Ensure the output directory exists.
    if not os.path.exists(PREPROCESSED_DIR):
        os.makedirs(PREPROCESSED_DIR)
    
    # Write the combined preprocessed patch notes to a single file.
    output_path = os.path.join(PREPROCESSED_DIR, "combined_preprocessed_patch_notes.json")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(unique_notes, outfile, ensure_ascii=False, indent=4)
    
    # --- Create a histogram of the number of patch notes per game ---
    # Load the combined preprocessed notes as a DataFrame.
    df = load_preprocessed_notes()
    
    # Group by game identifier ('appid') and count the patch notes.
    patchnotes_counts = df.groupby('appid').size()
    
    # Plot a histogram that shows how many games have a given number of patch notes.
    plt.figure(figsize=(10, 6))
    bins = range(patchnotes_counts.min(), patchnotes_counts.max() + 2)
    plt.hist(patchnotes_counts, bins=bins, edgecolor='black')
    plt.xlabel("Number of Patch Notes")
    plt.ylabel("Number of Games")
    plt.title("Histogram: Number of Patch Notes per Game")
    plt.tight_layout()
    plt.show()
