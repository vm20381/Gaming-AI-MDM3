import os
import requests
import json
import nltk
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

def load_patch_notes(directory='patch_notes', appid=None, return_all=False):
    """
    Load patch notes from JSON files in the specified directory.

    :param directory: Directory where JSON files are stored.
    :param appid: The AppID of the game to fetch patch notes for (optional).
    :param return_all: If True, fetch patch notes for all games (default is False).
    :return: List of patch notes.
    """
    all_patch_notes = []
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('_patch_notes.json'):
            file_appid = filename.split('_')[0]  # Extract AppID from filename
            
            # If specific AppID is provided, only load that file
            if appid and file_appid != str(appid):
                continue
            
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                patch_notes = json.load(file)
                all_patch_notes.extend(patch_notes)

            # If specific AppID was requested, return immediately after loading
            if appid:
                return all_patch_notes
    
    return all_patch_notes if return_all else []

all_notes = load_patch_notes(return_all=True)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define custom stopwords
custom_ignore_words = ['game', 'http', 'ubisoft', 'quot', 'csgo', 'strong', 'read', 'new', 'added', 'removed', 'noopener', 'nbsp', 'apos', 'valve', 'like', 'really', 'https', 'also', 'one', 'two', 'update', 'patch', 'steam', 'play', 'rust', 'dayz', 'pubg', 'apex', 'legends', 'team', 'fortress', '2', 'counter', 'strike', 'global', 'offensive', 'cs', 'go', 'rainbow', 'six', 'siege', 'delta', 'force', 'x', 'marvel', 'rivals']
stop_words = set(stopwords.words('english')).union(set(custom_ignore_words))

def load_patch_notes(directory='patch_notes', appid=None, return_all=False):
    """
    Load patch notes from JSON files in the specified directory.

    :param directory: Directory where JSON files are stored.
    :param appid: The AppID of the game to fetch patch notes for (optional).
    :param return_all: If True, fetch patch notes for all games (default is False).
    :return: List of patch notes.
    """
    all_patch_notes = []

    for filename in os.listdir(directory):
        if filename.endswith('_patch_notes.json'):
            file_appid = filename.split('_')[0] 
            
            if appid and file_appid != str(appid):
                continue

            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                patch_notes = json.load(file)
                all_patch_notes.extend(note['contents'] for note in patch_notes)

            if appid:
                return all_patch_notes
    
    return all_patch_notes if return_all else []

all_documents = load_patch_notes(appid=252490) # Load patch notes for Rust
# Preprocess documents
processed_docs = []
for doc in all_documents:
    tokens = word_tokenize(doc.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    processed_docs.append(tokens)

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Train LDA model
num_topics = 4
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)

# Display topics
print("Extracted Topics:")
topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)

# Visualize topics using pyLDAvis
vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda_visualization.html')
print("Interactive LDA visualization saved as 'lda_visualization.html'.")

# Generate word clouds for each topic
def generate_word_cloud(topic_id, num_words=50):
    topic_words = lda_model.show_topic(topic_id, num_words)
    word_freq = {word: prob for word, prob in topic_words}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for Topic {topic_id}")
    plt.show()

# for i in range(num_topics):
#     generate_word_cloud(i)

# PCA Analysis on LDA
def get_document_topic_matrix(lda_model, corpus):
    num_topics = lda_model.num_topics
    return np.array([sparse2full(lda_model[doc], num_topics) for doc in corpus])

doc_topic_matrix = get_document_topic_matrix(lda_model, corpus)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(doc_topic_matrix)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])

# Plot PCA Results
plt.figure(figsize=(10, 6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], alpha=0.5, c=np.argmax(doc_topic_matrix, axis=1), cmap="tab10")
plt.colorbar(label="Dominant Topic")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of LDA Topics")
plt.grid()
plt.show()

print("Explained Variance by PCA Components:", pca.explained_variance_ratio_)

# from sklearn.manifold import TSNE

# # Apply t-SNE to reduce to 2 dimensions
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# tsne_result = tsne.fit_transform(scaled_data)  # scaled_data from PCA preprocessing

# # Convert to DataFrame
# df_tsne = pd.DataFrame(tsne_result, columns=["TSNE1", "TSNE2"])

# # Plot t-SNE Results
# plt.figure(figsize=(10, 6))
# plt.scatter(df_tsne["TSNE1"], df_tsne["TSNE2"], alpha=0.5, c=np.argmax(doc_topic_matrix, axis=1), cmap="tab10")
# plt.colorbar(label="Dominant Topic")
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.title("t-SNE Visualization of LDA Topics")
# plt.grid()
# plt.show()
