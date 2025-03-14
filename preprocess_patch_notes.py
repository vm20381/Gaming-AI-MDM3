import os
import re
import json
import pandas as pd
from nltk.tokenize import word_tokenize

RAW_DIR = r"C:\Users\Orlan\Documents\MDM3\Gaming-AI\Gaming-AI-MDM3\datasets\patch_notes"
PREPROCESSED_DIR = r"C:\Users\Orlan\Documents\MDM3\Gaming-AI\Gaming-AI-MDM3\datasets\preprocessed_patch_notes"

CUSTOM_IGNORE_WORDS = [
    'steamdb', 'store', 'storeparser', 'storeparsercom', 'storeparsercomsteamdb',
    'patchnotes', 'tom', 'clancy', 'game', 'http', 'ubisoft', 'quot', 'csgo',
    'strong', 'read', 'new', 'added', 'removed', 'noopener', 'nbsp', 'apos',
    'valve', 'like', 'really', 'https', 'also', 'one', 'two', 'update', 'patch',
    'steam', 'play', 'rust', 'dayz', 'pubg', 'apex', 'legends', 'team',
    'fortress', '2', 'counter', 'strike', 'global', 'offensive', 'cs', 'go',
    'rainbow', 'six', 'siege', 'delta', 'force', 'x', 'marvel', 'rivals',
    'released', 'encouraged', 'STEAM_CLAN_IMAGE', 'battlegrounds', 'battleground', 
    'chicken', 'dinner', 'playerunknown', 'player', 'unknown', 'battle', 'royale', 
    'pubg', 'pubg', 'pubg', 'pubg', 'pubg', 'valorant', 'changes', 'games', 'gaming'
]

def load_preprocessed_notes(directory=PREPROCESSED_DIR, appid=None, return_all=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(base_dir, directory)
        
    all_notes = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_appid = filename.split('.')[0]
            if appid is not None and file_appid != str(appid):
                continue
            
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as infile:
                notes = json.load(infile)
                all_notes.extend(notes)
            
            if appid is not None:
                break

    df = pd.DataFrame(all_notes, columns=['content', 'gid', 'date', 'appid'])
    return df

def preprocess_doc(doc):
    text = doc.get('contents', '')
    # Remove URLs
    text = ' '.join(word for word in text.split() if 'http' not in word)
    # Remove HTML tags
    text = text.replace("&nbsp;", " ").replace("&apos;", "'").replace("&quot;", '"')
    
    # Tokenize once and process tokens
    tokens = word_tokenize(text)
    
    # Lowercase words if they are in CUSTOM_IGNORE_WORDS
    tokens = [word.lower() if word.lower() in CUSTOM_IGNORE_WORDS else word for word in tokens]
    
    # Remove any Custom Ignore Words    
    tokens = [word for word in tokens if word not in CUSTOM_IGNORE_WORDS]
    
    # Keep only alphabetic tokens or punctuation (.,!,?,)
    tokens = [word for word in tokens if word.isalpha() or word in ['.', ',', '!', '?']]
    
    # Join tokens with a space then remove any space before punctuation
    text = ' '.join(tokens)
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    
    return text

if __name__ == "__main__":
    for filename in os.listdir(RAW_DIR):
        if filename.endswith("_patch_notes.json"):
            input_path = os.path.join(RAW_DIR, filename)
            # Extract app id from filename (e.g., "221100_patch_notes.json")
            file_appid = filename.split('_')[0]
            output_filename = f"{file_appid}.json"  # now just the app id with .json
            output_path = os.path.join(PREPROCESSED_DIR, output_filename)
            
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
            
            # Remove duplicate notes based on the tuple (content, gid, date)
            unique_notes = []
            seen = set()
            for note in preprocessed_notes:
                identifier = (note['content'], note['gid'], note.get('date'))
                if identifier not in seen:
                    seen.add(identifier)
                    unique_notes.append(note)
            preprocessed_notes = unique_notes

            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(preprocessed_notes, outfile, ensure_ascii=False, indent=4)
            
            print(f"Saved {len(preprocessed_notes)} preprocessed patch notes to {output_filename}")
