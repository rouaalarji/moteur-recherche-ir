
import time
import pickle
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Fichiers d'index et métadonnées
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "..", "data", "index.pkl")
META_PATH = os.path.join(BASE_DIR, "..", "data", "meta.pkl")
DOCS_DIR = os.path.join(BASE_DIR, "..", "data", "documents")

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

# Recherche avec extrait
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
        # Charger le contenu pour extrait
        doc_path = os.path.join(DOCS_DIR, f"doc_{meta[idx]['id']:03d}.json")
        excerpt = ""
        if os.path.exists(doc_path):
            with open(doc_path, encoding="utf-8") as f:
                content = json.load(f)
            excerpt = content.get("content", "")[:200] + "..."
        results.append({
            "title": meta[idx]["title"],
            "url": meta[idx]["url"],
            "score": round(float(scores[idx]), 4),
            "excerpt": excerpt
        })
    return results

# Interface CLI
def main():
    print("\n=== Moteur de Recherche CLI ===")
    print("Tapez votre requête (ou 'exit' pour quitter)\n")
    while True:
        query = input(" Requête: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("\n Au revoir!")
            break
        start_time = time.time()
        results = search(query, top_k=5)
        elapsed = time.time() - start_time
        print(f"\n Temps de recherche: {elapsed:.3f} sec\n")
        for i, r in enumerate(results, start=1):
            print(f"{i}. {r['title']} (score={r['score']})\n   URL: {r['url']}\n   Extrait: {r['excerpt']}\n")

if __name__ == "__main__":
    main()
