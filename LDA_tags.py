import os
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

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Custom stopwords
custom_ignore_words = { 'game', 'http', 'quot', 'new', 'added', 'removed', 'update', 'patch',
                        'steam', 'play', 'team', 'global', 'offensive', 'x', 'rivals'}
stop_words = set(stopwords.words('english')).union(custom_ignore_words)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]

def load_game_tags(filename):
    df = pd.read_csv(filename)
    df['Tags'] = df['Tags'].fillna('')  # Handle missing tags
    return df

def process_game_tags(dataset):
    processed_docs = [tags.split(', ') for tags in dataset['Tags']]
    return processed_docs

def train_lda(processed_docs, num_topics=4):
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)
    return lda_model, corpus, dictionary

def visualize_lda(lda_model, corpus, dictionary, output_file='lda_visualization.html'):
    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, output_file)
    print(f"Interactive LDA visualization saved as '{output_file}'")

def generate_word_cloud(lda_model, topic_id, num_words=50):
    topic_words = lda_model.show_topic(topic_id, num_words)
    word_freq = {word: prob for word, prob in topic_words}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for Topic {topic_id}")
    plt.show()

def apply_pca(lda_model, corpus):
    doc_topic_matrix = np.array([sparse2full(lda_model[doc], lda_model.num_topics) for doc in corpus])
    scaled_data = StandardScaler().fit_transform(doc_topic_matrix)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df_pca["PC1"], df_pca["PC2"], alpha=0.5, c=np.argmax(doc_topic_matrix, axis=1), cmap="tab10")
    plt.colorbar(label="Dominant Topic")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Visualization of LDA Topics")
    plt.grid()
    plt.show()
    
    print("Explained Variance by PCA Components:", pca.explained_variance_ratio_)

if __name__ == "__main__":
    dataset = load_game_tags("C:/Uni/MDM3/gaming/datasets/game_data.csv")
    processed_docs = process_game_tags(dataset)
    lda_model, corpus, dictionary = train_lda(processed_docs)
    visualize_lda(lda_model, corpus, dictionary)
    apply_pca(lda_model, corpus)
    
    # Generate word clouds for topics
    for i in range(lda_model.num_topics):
        generate_word_cloud(lda_model, i)