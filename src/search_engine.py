import os
import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi

# Télécharger les stopwords
nltk.download("stopwords")
french_stopwords = set(stopwords.words("french"))

# Dossiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "..", "data", "index.pkl")
META_PATH = os.path.join(BASE_DIR, "..", "data", "meta.pkl")
DOCS_DIR = os.path.join(BASE_DIR, "..", "data", "documents")


# -----------------------------------------------------------
# Nettoyage
# -----------------------------------------------------------
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\sàâçéèêëîïôûùüÿœ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    tokens = normalize_text(text).split()
    tokens = [t for t in tokens if t not in french_stopwords]
    return tokens


# -----------------------------------------------------------
# Charger l'index
# -----------------------------------------------------------
def load_index():
    with open(INDEX_PATH, "rb") as f:
        index_data = pickle.load(f)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index_data, meta


# -----------------------------------------------------------
# Fonction de recherche (TF-IDF ou BM25)
# -----------------------------------------------------------
def search(query: str, top_k: int = 5, model: str = "tfidf"):
    index_data, meta = load_index()

    vectorizer = index_data.get("vectorizer")
    tfidf_matrix = index_data.get("tfidf")
    tokenized_docs = index_data.get("tokenized_docs")
    bm25 = index_data.get("bm25")
    #Transforme la requête en minuscules, supprime la ponctuation et les espaces multiples.
    clean_query = normalize_text(query)

    results = []

    # ---------------- TF-IDF ---------------- #
    if model == "tfidf":
        query_vec = vectorizer.transform([clean_query]) #transforme la requête en vecteur TF-IDF
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        for idx in ranked_indices:
            results.append({
                "title": meta[idx]["title"],
                "url": meta[idx]["url"],
                "score": round(float(scores[idx]), 4),
                "id": meta[idx]["id"]
            })

    # ---------------- BM25 ---------------- #
    elif model == "bm25":
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = bm25.get_scores(q_tokens)
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        for idx in ranked_indices:
            results.append({
                "title": meta[idx]["title"],
                "url": meta[idx]["url"],
                "score": round(float(scores[idx]), 4),
                "id": meta[idx]["id"]
            })

    else:
        raise ValueError("model must be either 'tfidf' or 'bm25'")

    return results


# -----------------------------------------------------------
# Test en ligne de commande
# -----------------------------------------------------------
if __name__ == "__main__":
    q = input(" Entrez votre requête : ")

    print("\n--- Résultats TF-IDF ---")
    for r in search(q, top_k=5, model="tfidf"):
        print(r)

    print("\n--- Résultats BM25 ---")
    for r in search(q, top_k=5, model="bm25"):
        print(r)
