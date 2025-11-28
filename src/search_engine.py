
import os
import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Dossiers et fichiers 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "..", "data", "index.pkl")
META_PATH = os.path.join(BASE_DIR, "..", "data", "meta.pkl")

# Charger l'index et les métadonnées
def load_index():
    with open(INDEX_PATH, "rb") as f:
        index_data = pickle.load(f)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index_data["vectorizer"], index_data["tfidf"], meta

# Prétraiter la requête 
def preprocess_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r"[^\w\sàâçéèêëîïôûùüÿœ]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query

# Calculer la similarité cosinus et retourner top-K résultats
def search(query: str, top_k: int = 5):
    vectorizer, tfidf_matrix, meta = load_index()
    clean_query = preprocess_query(query)
    query_vec = vectorizer.transform([clean_query])

    # Similarité cosinus
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Trier par score décroissant
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in ranked_indices:
        results.append({
            "title": meta[idx]["title"],
            "url": meta[idx]["url"],
            "score": round(float(scores[idx]), 4)
        })
    return results

# Exemple d'utilisation
def main():
    query = input(" Entrez votre requête: ")
    print(f"\nRésultats pour: '{query}'\n")
    results = search(query, top_k=5)
    for i, r in enumerate(results, start=1):
        print(f"{i}. {r['title']} (score={r['score']})\n   URL: {r['url']}\n")

if __name__ == "__main__":
    main()
