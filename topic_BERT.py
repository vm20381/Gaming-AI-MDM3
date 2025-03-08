from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
import umap.umap_ as umap_module
from preprocess_patch_notes import load_preprocessed_notes

# Attempt to import cuML's UMAP and HDBSCAN for GPU acceleration
try:
    from cuml.manifold import UMAP as cumlUMAP
    from cuml.cluster import HDBSCAN as cumlHDBSCAN
    gpu_available = True
except ImportError:
    gpu_available = False

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

# Set your APP_ID and return_all flag
APP_ID = 221100
RETURN_ALL = False

# Load preprocessed notes
processed_docs = load_preprocessed_notes(return_all=RETURN_ALL) if RETURN_ALL else load_preprocessed_notes(appid=APP_ID)
print(f"Loaded {len(processed_docs)} preprocessed patch notes")

if not processed_docs:
    print("No preprocessed patch notes were found. Please run the preprocessor to generate them.")
    exit(1)

# Initialize embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

# Initialize BERTopic with appropriate UMAP and HDBSCAN models
if gpu_available:
    print("cuML is available. Using GPU-accelerated UMAP and HDBSCAN.")
    custom_umap = cumlUMAP(n_neighbors=50, n_components=5, metric="cosine", init="random", random_state=42)
    custom_hdbscan = cumlHDBSCAN(min_samples=20, gen_min_span_tree=True, prediction_data=True)
else:
    print("cuML is not available. Using CPU-based UMAP and HDBSCAN with CPU parallelization.")
    custom_umap = umap.UMAP(n_neighbors=50, n_components=5, metric="cosine", init="random", random_state=42)
    from hdbscan import HDBSCAN
    # Enable parallel computation of core distances by setting core_dist_n_jobs=-1
    custom_hdbscan = HDBSCAN(min_samples=20, gen_min_span_tree=True, prediction_data=True, core_dist_n_jobs=5)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=custom_umap,
    hdbscan_model=custom_hdbscan,
    language="english",
    nr_topics=None,
    calculate_probabilities=True,
    verbose=True,
)

topics, probabilities = topic_model.fit_transform(processed_docs)

# Save results to CSV
csv_path = "all_results.csv" if RETURN_ALL else f"{APP_ID}_results.csv"
topic_info_df = topic_model.get_topic_info().set_index("Topic")

with open(csv_path, 'w', encoding='utf-8') as f:
    f.write("Topic,Topic_Name,Probability\n")
    for topic, prob in zip(topics, probabilities):
        topic_name = topic_info_df.loc[topic, "Name"] if topic in topic_info_df.index else ""
        f.write(f"{topic},{topic_name},{prob}\n")

print(f"Results saved to {csv_path}")

# Display topic information
topic_info = topic_model.get_topic_info()
print(topic_info)

for topic_id in topic_info['Topic'].unique():
    if topic_id != -1:
        print(f"Topic {topic_id}:")
        print(topic_model.get_topic(topic_id))
        print("\n")

# Visualize topics
fig = topic_model.visualize_topics()
fig.show()
