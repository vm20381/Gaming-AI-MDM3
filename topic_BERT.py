import os 
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bertopic import BERTopic
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import umap
import umap.umap_ as umap_module
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from steamWebAPI_scrape import load_patch_notes

# Patch UMAP spectral_layout if needed
orig_spectral_layout = umap_module.spectral_layout
def patched_spectral_layout(data, graph, n_components, random_state, **kwargs):
    N = graph.shape[0]
    if (n_components + 1) >= N:
        A = graph.toarray() if hasattr(graph, "toarray") else graph
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(A)
        return eigenvectors[:, :n_components]
    else:
        return orig_spectral_layout(data, graph, n_components, random_state, **kwargs)
umap_module.spectral_layout = patched_spectral_layout

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

custom_ignore_words = [
    'steamdb', 'store', 'storeparser', 'storeparsercom', 'storeparsercomsteamdb',
    'patchnotes', 'tom', 'clancy', 'game', 'http', 'ubisoft', 'quot', 'csgo',
    'strong', 'read', 'new', 'added', 'removed', 'noopener', 'nbsp', 'apos',
    'valve', 'like', 'really', 'https', 'also', 'one', 'two', 'update', 'patch',
    'steam', 'play', 'rust', 'dayz', 'pubg', 'apex', 'legends', 'team',
    'fortress', '2', 'counter', 'strike', 'global', 'offensive', 'cs', 'go',
    'rainbow', 'six', 'siege', 'delta', 'force', 'x', 'marvel', 'rivals',
    'released', 'encouraged', 'automatically'
]
stop_words = set(stopwords.words('english')).union(set(custom_ignore_words))

# Set your app ID and return_all flag.
APP_ID = 221100
RETURN_ALL = True  # Set to True to load all patch notes

if RETURN_ALL:
    all_documents = load_patch_notes(return_all=True)
else:
    all_documents = load_patch_notes(appid=APP_ID)

print(f"Loaded {len(all_documents)} patch notes")

lemmatizer = WordNetLemmatizer()
def preprocess_doc(doc):
    if isinstance(doc, dict):
        doc = doc.get('contents', '')
    tokens = word_tokenize(doc.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    bigrams = ['_'.join(gram) for gram in ngrams(tokens, 2)]
    trigrams = ['_'.join(gram) for gram in ngrams(tokens, 3)]
    all_tokens = tokens + bigrams + trigrams
    return " ".join(all_tokens)

processed_docs = []
for doc in all_documents:
    processed_doc = preprocess_doc(doc)
    if processed_doc.strip():
        processed_docs.append(processed_doc)

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

topic_model = BERTopic(embedding_model=embedding_model, language="english", nr_topics=5,
                       calculate_probabilities=True, verbose=True)
topics, probabilities = topic_model.fit_transform(processed_docs)

# Determine CSV file name based on parameters:
if RETURN_ALL:
    csv_path = "all_results.csv"
else:
    csv_path = f"{APP_ID}_results.csv"

# Retrieve topic info to get topic names
topic_info_df = topic_model.get_topic_info().set_index("Topic")

with open(csv_path, 'w', encoding='utf-8') as f:
    f.write("Topic,Topic_Name,Probability\n")
    for topic, prob in zip(topics, probabilities):
         # Lookup the topic name if available, else leave blank
         topic_name = topic_info_df.loc[topic, "Name"] if topic in topic_info_df.index else ""
         f.write(f"{topic},{topic_name},{prob}\n")

print(f"Results saved to {csv_path}")

topic_info = topic_model.get_topic_info()
print(topic_info)

for topic_id in topic_info['Topic'].unique():
    if topic_id != -1:
        print(f"Topic {topic_id}:")
        print(topic_model.get_topic(topic_id))
        print("\n")

custom_umap = umap.UMAP(n_neighbors=2, n_components=2, metric="cosine", init="random", random_state=42)
topic_model.umap_model = custom_umap

fig = topic_model.visualize_topics()
fig.show()
