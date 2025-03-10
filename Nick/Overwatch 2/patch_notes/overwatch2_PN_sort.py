import json
import pandas as pd
from datetime import datetime

# Load patch notes JSON
with open('2357570_patch_notes.json', 'r', encoding='utf-8') as f:
    patch_notes = json.load(f)

# Convert to DataFrame
df_patches = pd.DataFrame(patch_notes)

# Convert Unix timestamps to human-readable dates
df_patches['readable_date'] = df_patches['date'].apply(
    lambda ts: datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
)

# Optionally, extract month and year for merging later
df_patches['Year'] = pd.to_datetime(df_patches['readable_date']).dt.year
df_patches['Month'] = pd.to_datetime(df_patches['readable_date']).dt.month

print(df_patches[['title', 'readable_date']].head())
