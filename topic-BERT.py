import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bertopic import BERTopic
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import umap

GPU_UMAP = umap.UMAP
use_gpu_umap = False
print("cuML not available. Using CPU-based UMAP.")

# (Optional) Patch spectral_layout only if needed for CPU UMAP
if not use_gpu_umap:
    import umap.umap_ as umap_module
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

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define custom stopwords
custom_ignore_words = ['steamdb', 'store', 'storeparser', 'storeparsercom', 'storeparsercomsteamdb', 'patchnotes', 'tom', 'clancy',
    'game', 'http', 'ubisoft', 'quot', 'csgo', 'strong', 'read', 'new', 'added', 'removed',
    'noopener', 'nbsp', 'apos', 'valve', 'like', 'really', 'https', 'also', 'one', 'two',
    'update', 'patch', 'steam', 'play', 'rust', 'dayz', 'pubg', 'apex', 'legends', 'team',
    'fortress', '2', 'counter', 'strike', 'global', 'offensive', 'cs', 'go', 'rainbow', 'six',
    'siege', 'delta', 'force', 'x', 'marvel', 'rivals'
]
stop_words = set(stopwords.words('english')).union(set(custom_ignore_words))

def load_patch_notes(directory='patch_notes', appid=None, return_all=False):
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

all_documents = load_patch_notes(appid=730)

processed_docs = []
for doc in all_documents:
    tokens = word_tokenize(doc.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    processed_docs.append(" ".join(tokens))

# Load the SentenceTransformer model on GPU
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

# Instantiate BERTopic with the GPU-based embedding model
topic_model = BERTopic(embedding_model=embedding_model, language="english", nr_topics=3)

topics, probabilities = topic_model.fit_transform(processed_docs)

topic_info = topic_model.get_topic_info()
print(topic_info)

# Display the top words for each topic (skipping the outlier topic -1)
for topic_id in topic_info['Topic'].unique():
    if topic_id != -1:
        print(f"Topic {topic_id}:")
        print(topic_model.get_topic(topic_id))
        print("\n")

# Set up UMAP using GPU acceleration if available
custom_umap = GPU_UMAP(n_neighbors=2, n_components=2, metric="cosine", init="random", random_state=42)
topic_model.umap_model = custom_umap

fig = topic_model.visualize_topics()
fig.show()
