import os 
import sys
import pickle
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired
from preprocess_patch_notes import load_preprocessed_notes

APP_ID = 221100
RETURN_ALL = True

# File paths for saving/loading
MODEL_PATH = r"C:\Users\Orlan\Documents\MDM3\Gaming-AI\Gaming-AI-MDM3\saved_bertopic_model"
EMBEDDINGS_PATH = r"C:\Users\Orlan\Documents\MDM3\Gaming-AI\Gaming-AI-MDM3\saved_embeddings.pkl"

def run_topic_model_fitting(docs, embedding_model):
    embeddings = embedding_model.encode(docs, show_progress_bar=True)
    representation_model = KeyBERTInspired()
    custom_umap = umap.UMAP(n_neighbors=10, n_components=5, metric="cosine", init="random", random_state=42)
    custom_hdbscan = HDBSCAN(min_samples=20, gen_min_span_tree=True, prediction_data=True, core_dist_n_jobs=5)

    topic_model = BERTopic(
        representation_model=representation_model,
        embedding_model=embedding_model,
        umap_model=custom_umap,
        hdbscan_model=custom_hdbscan,
        language="english",
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True,
    )
    
    topics, probs = topic_model.fit_transform(documents=docs, embeddings=embeddings)

    topic_model.save(MODEL_PATH)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    
    return topic_model, embeddings, topics, probs

def main():
    # Load preprocessed notes as a DataFrame with 'content', 'gid', and 'date'
    data = load_preprocessed_notes(return_all=RETURN_ALL) if RETURN_ALL else load_preprocessed_notes(appid=APP_ID)
    print(f"Loaded {len(data)} preprocessed patch notes")
    
    # Extract docs (for modeling), gids (for keying), and dates
    docs = data['content'].tolist()
    gids = data['gid'].tolist()
    dates = data['date'].tolist()
    appId = data['appid'].tolist()

    if '-y' not in sys.argv:
        user_input = input("Continue analysis with these patch notes? (yes/no): ")
        if user_input.lower().strip() not in ['yes', 'y']:
            print("Exiting analysis.")
            return

    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

    if os.path.exists(MODEL_PATH) and os.path.exists(EMBEDDINGS_PATH):
        load_choice = input("Saved topic model and embeddings found. Load them? (yes/no): ")
        if load_choice.lower().strip() in ['yes', 'y']:
            print("Loading saved embeddings and topic model...")
            with open(EMBEDDINGS_PATH, "rb") as f:
                embeddings = pickle.load(f)

            topic_model = BERTopic.load(MODEL_PATH)
            topics, probs = topic_model.transform(documents=docs, embeddings=embeddings)
        else:
            print("Re-running the fitting process...")
            topic_model, embeddings, topics, probs = run_topic_model_fitting(docs, embedding_model)
    else:
        print("No saved model/embeddings found. Running the fitting process...")
        topic_model, embeddings, topics, probs = run_topic_model_fitting(docs, embedding_model)

    print("Before Topic reduction", topic_model.get_topic_info())
    topic_model.reduce_topics(docs, nr_topics=21)
    topics = topic_model.topics_

    print("Before outlier reduction", topic_model.get_topic_info())
    topics = topic_model.reduce_outliers(docs, topics)

    print("Final topics", topic_model.get_topic_info())

    results_df = pd.DataFrame({'gid': gids, 'topic': topics, 'date': dates, 'appid': appId})
    csv_path = "all_results.csv" if RETURN_ALL else f"{APP_ID}_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    fig = topic_model.visualize_barchart()
    fig.show()

    fig = topic_model.visualize_topics()
    fig.show()

if __name__ == "__main__":
    main()
