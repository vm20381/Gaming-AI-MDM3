import os
import json
from nltk.tokenize import word_tokenize

CUSTOM_IGNORE_WORDS = [
    'steamdb', 'store', 'storeparser', 'storeparsercom', 'storeparsercomsteamdb',
    'patchnotes', 'tom', 'clancy', 'game', 'http', 'ubisoft', 'quot', 'csgo',
    'strong', 'read', 'new', 'added', 'removed', 'noopener', 'nbsp', 'apos',
    'valve', 'like', 'really', 'https', 'also', 'one', 'two', 'update', 'patch',
    'steam', 'play', 'rust', 'dayz', 'pubg', 'apex', 'legends', 'team',
    'fortress', '2', 'counter', 'strike', 'global', 'offensive', 'cs', 'go',
    'rainbow', 'six', 'siege', 'delta', 'force', 'x', 'marvel', 'rivals',
    'released', 'encouraged', 'automatically'
]

def load_preprocessed_notes(directory="preprocessed_patch_notes", appid=None, return_all=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(base_dir, directory)
    
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return []
    
    all_notes = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            # Now file names are like "221100.json"
            file_appid = filename.split('.')[0]
            if appid is not None and file_appid != str(appid):
                continue
            
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as infile:
                notes = json.load(infile)
                all_notes.extend(notes)
            
            if appid is not None:
                return all_notes
    return all_notes if return_all else []

def preprocess_doc(doc):
    if isinstance(doc, dict):
        text = doc.get('contents', '')
    else:
        text = doc
    tokens = word_tokenize(text.lower())
    return " ".join(tokens)

RAW_DIR = "datasets/patch_notes"
PREPROCESSED_DIR = "datasets/preprocessed_patch_notes"

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
                preprocessed_notes.append(preprocessed_text)
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(preprocessed_notes, outfile, ensure_ascii=False, indent=4)
        
        print(f"Saved {len(preprocessed_notes)} preprocessed patch notes to {output_filename}")
