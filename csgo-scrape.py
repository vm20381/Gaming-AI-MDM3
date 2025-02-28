import os
import requests
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import time
from scipy.stats import zscore

# Download NLTK stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Define the base URLs for the Steam API
url_news = "http://api.steampowered.com/ISteamNews/GetNewsForApp/v0002/"
url_player_data = "http://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"

# Parameters for the API request for news
params_news = {
    "appid": 730,        # CS:GO App ID
    "count": 20,         # Number of news posts per request (max limit)
    "maxlength": 3000,    # Maximum length of each post
    "format": "json"     # The format of the response
}

# Parameters for the API request for player base
params_player = {
    "appid": 730,        # CS:GO App ID
    "format": "json"     # The format of the response
}

# Filenames for caching API results
PATCH_NOTES_FILE = 'csgo_patch_notes.json'
PLAYER_BASE_FILE = 'csgo_player_base.json'

# Function to estimate the total number of news items
def estimate_total_news_items():
    start = 1000  # Arbitrary large start value
    params_news["start"] = start
    response = requests.get(url_news, params=params_news)
    if response.status_code == 200:
        data = response.json()
        if data['appnews']['newsitems']:
            return start + len(data['appnews']['newsitems'])
    return 0  # Return 0 if no news items are found

# Function to fetch all patch notes from the API
def fetch_all_patch_notes():
    patch_notes = []
    total_news_items = estimate_total_news_items()
    if total_news_items == 0:
        print("No news items found.")
        return patch_notes

    # Fetch news items in batches
    for start in range(0, total_news_items, params_news["count"]):
        params_news["start"] = start
        response = requests.get(url_news, params=params_news)
        if response.status_code == 200:
            data = response.json()
            if data['appnews']['newsitems']:
                patch_notes.extend(data['appnews']['newsitems'])
            else:
                break  # No more news items
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            break

    return patch_notes

# Function to fetch player base data from the API (sampled)
def fetch_player_base(sample_interval=5):
    player_base_data = []
    for i, date in enumerate(dates):
        if i % sample_interval == 0:
            player_count = fetch_player_base_single()
            if player_count is not None:
                player_base_data.append({'date': date, 'player_count': player_count})
            time.sleep(1)  # Pause to avoid hitting the API rate limit
    return player_base_data

# Function to fetch a single player base measurement from the API
def fetch_player_base_single():
    response = requests.get(url_player_data, params=params_player)
    if response.status_code == 200:
        data = response.json()
        if 'response' in data and 'player_count' in data['response']:
            return data['response']['player_count']
    return None

# ------------------- Data Loading / Caching -------------------

# Load or fetch patch notes
if os.path.exists(PATCH_NOTES_FILE):
    with open(PATCH_NOTES_FILE, 'r', encoding='utf-8') as f:
        patch_notes = json.load(f)
    print(f"Loaded {len(patch_notes)} patch notes from '{PATCH_NOTES_FILE}'")
else:
    patch_notes = fetch_all_patch_notes()
    with open(PATCH_NOTES_FILE, 'w', encoding='utf-8') as f:
        json.dump(patch_notes, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(patch_notes)} patch notes to '{PATCH_NOTES_FILE}'")

# Extract the 'contents' and 'date' of each news item
documents = [note['contents'] for note in patch_notes]
dates = [note['date'] for note in patch_notes]

# Load or fetch player base data
if os.path.exists(PLAYER_BASE_FILE):
    df_player_base = pd.read_json(PLAYER_BASE_FILE)
    print(f"Loaded player base data from '{PLAYER_BASE_FILE}'")
else:
    # Use a sample interval to reduce the number of API calls
    player_base_data = fetch_player_base(sample_interval=5)
    df_player_base = pd.DataFrame(player_base_data)
    # Save the player base data for future use
    df_player_base.to_json(PLAYER_BASE_FILE, orient='records', date_format='iso')
    print(f"Saved player base data to '{PLAYER_BASE_FILE}'")

# Convert the date from string to datetime (assuming Unix epoch seconds)
df_player_base['date'] = pd.to_datetime(df_player_base['date'], unit='s')

# Print summary statistics for the player base data
print("Summary statistics for player base data:")
print(df_player_base['player_count'].describe())

# If you want to see the first few rows as well:
print("\nFirst few rows of player base data:")
print(df_player_base.head())

# ------------------- Preprocessing and LDA -------------------

# Tokenize and preprocess the text data
stop_words = set(stopwords.words('english'))
processed_docs = []
for doc in documents:
    tokens = word_tokenize(doc.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    processed_docs.append(tokens)

# Combine all tokens from all documents into one list
all_tokens = [token for doc in processed_docs for token in doc]

# Compute the frequency distribution of words
freq_dist = nltk.FreqDist(all_tokens)

# Plot the histogram for the top 30 most common words
plt.figure(figsize=(12, 6))
freq_dist.plot(100, title="Word Frequency Histogram", cumulative=False)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()

# Create a dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Apply LDA to the corpus
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=30)

# Display the topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)

# Calculate the topic distribution for each document
doc_topics = [lda_model.get_document_topics(doc) for doc in corpus]
doc_topic_matrix = np.array([[dict(doc).get(i, 0) for i in range(lda_model.num_topics)] for doc in doc_topics])
dominant_topics = np.argmax(doc_topic_matrix, axis=1)

# For simplicity, assume that the player base data corresponds to the patch notes in order
df_player_base = df_player_base.iloc[:len(dominant_topics)].copy()
df_player_base['dominant_topic'] = dominant_topics[:len(df_player_base)]

# ------------------- Statistical Analysis -------------------

mean_player_count = df_player_base['player_count'].mean()
std_player_count = df_player_base['player_count'].std()
df_player_base['z_score'] = (df_player_base['player_count'] - mean_player_count) / std_player_count
threshold = 2
df_player_base['significant_change'] = df_player_base['z_score'].abs() > threshold

topic_stats = df_player_base.groupby('dominant_topic').agg(
    avg_z=('z_score', 'mean'),
    count_significant=('significant_change', 'sum'),
    total_updates=('z_score', 'count')
).reset_index()
topic_stats['percent_significant'] = topic_stats['count_significant'] / topic_stats['total_updates'] * 100
topic_stats = topic_stats.sort_values(by='avg_z', key=lambda x: x.abs(), ascending=False)

print("\nRanking of Topics by Average Z-Score and Significant Changes:")
print(topic_stats)

# ------------------- Visualizations -------------------

# 1. Plot just the player base over time
plt.figure(figsize=(10, 6))
plt.plot(df_player_base['date'], df_player_base['player_count'], color='blue', marker='o')
plt.title('Player Base Over Time')
plt.xlabel('Date')
plt.ylabel('Player Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Plot player base over time with dominant topics highlighted
plt.figure(figsize=(10, 6))
plt.plot(df_player_base['date'], df_player_base['player_count'], label='Player Count', color='b')
for topic_num in range(lda_model.num_topics):
    topic_changes = df_player_base[df_player_base['dominant_topic'] == topic_num]
    plt.scatter(topic_changes['date'], topic_changes['player_count'], label=f"Topic {topic_num}", s=30)
plt.title('Player Base Over Time with LDA Topics')
plt.xlabel('Date')
plt.ylabel('Player Count')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(title='Topics')
plt.tight_layout()
plt.show()

# # 3. Plot average Z-score per topic
# plt.figure(figsize=(10, 6))
# plt.bar(topic_stats['dominant_topic'].astype(str), topic_stats['avg_z'], color='skyblue')
# plt.xlabel("Dominant Topic")
# plt.ylabel("Average Z-Score")
# plt.title("Average Z-Score of Player Count Changes by Dominant Topic")
# plt.axhline(0, color='black', linewidth=0.5)
# plt.show()

# # 4. Plot count of significant changes per topic
# plt.figure(figsize=(10, 6))
# plt.bar(topic_stats['dominant_topic'].astype(str), topic_stats['count_significant'], color='salmon')
# plt.xlabel("Dominant Topic")
# plt.ylabel("Count of Significant Changes")
# plt.title("Count of Statistically Significant Player Count Changes by Dominant Topic")
# plt.show()

# 5. pyLDAvis: Interactive Topic Visualization
vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)

# 6. Word Clouds for each topic
def generate_word_cloud(topic_id, num_words=50):
    topic_words = lda_model.show_topic(topic_id, num_words)
    word_freq = {word: prob for word, prob in topic_words}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for Topic {topic_id}")
    plt.show()

for i in range(lda_model.num_topics):
    generate_word_cloud(i)

# # 7. t-SNE Visualization of Document Topics
# doc_topic_dense = np.array([[dict(doc).get(i, 0) for i in range(lda_model.num_topics)] for doc in doc_topics])
# tsne = TSNE(n_components=2, random_state=0)
# doc_topic_tsne = tsne.fit_transform(doc_topic_dense)
# plt.figure(figsize=(10, 8))
# plt.scatter(doc_topic_tsne[:, 0], doc_topic_tsne[:, 1], alpha=0.5)
# plt.title('t-SNE Visualization of Document Topics')
# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# plt.show()
