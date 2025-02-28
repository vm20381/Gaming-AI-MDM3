import feedparser
import json

# Define the RSS feed URL
rss_url = "https://steamdb.info/api/PatchnotesRSS/?appid=730"

# Parse the RSS feed
feed = feedparser.parse(rss_url)

# Extract patch notes
patch_notes = []
for entry in feed.entries:
    patch = {
        "title": entry.title,
        "link": entry.link,
        "description": entry.description,
        "pubDate": entry.published
    }
    patch_notes.append(patch)

# Save the patch notes to a JSON file
with open('RSS_cs2_patch_notes.json', 'w', encoding='utf-8') as f:
    json.dump(patch_notes, f, ensure_ascii=False, indent=4)

print(f"Saved {len(patch_notes)} patch notes to 'cs2_patch_notes.json'")
