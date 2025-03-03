from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
import umap.umap_ as umap_module
from preprocess_patch_notes import load_preprocessed_notes  # updated import

# Patch UMAP's spectral_layout
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

# Set your APP_ID and return_all flag.
APP_ID = 221100
RETURN_ALL = True  # Set True to load all preprocessed notes

if RETURN_ALL:
    processed_docs = load_preprocessed_notes(return_all=True)
else:
    processed_docs = load_preprocessed_notes(appid=APP_ID)

print(f"Loaded {len(processed_docs)} preprocessed patch notes")

if not processed_docs:
    print("No preprocessed patch notes were found. Please run the preprocessor to generate them.")
    exit(1)

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

topic_model = BERTopic(embedding_model=embedding_model, language="english", nr_topics=5,
                       calculate_probabilities=True, verbose=True)
topics, probabilities = topic_model.fit_transform(processed_docs)

if RETURN_ALL:
    csv_path = "all_results.csv"
else:
    csv_path = f"{APP_ID}_results.csv"

topic_info_df = topic_model.get_topic_info().set_index("Topic")

with open(csv_path, 'w', encoding='utf-8') as f:
    f.write("Topic,Topic_Name,Probability\n")
    for topic, prob in zip(topics, probabilities):
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
