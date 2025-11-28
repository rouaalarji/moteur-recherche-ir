
import os
import json
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

# Télécharger les stopwords si nécessaire
nltk.download('stopwords')
french_stopwords = stopwords.words('french')

# Dossiers et fichiers
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "documents")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "index.pkl")
META_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "meta.pkl")

def load_documents():
    docs, meta = [], []
    files = sorted(f for f in os.listdir(DOCS_DIR) if f.endswith(".json"))
    for fname in files:
        with open(os.path.join(DOCS_DIR, fname), encoding="utf-8") as f:
            d = json.load(f)
            text = d.get("content", "")
            # Nettoyage simple
            text = text.lower()
            text = re.sub(r"[^\w\sàâçéèêëîïôûùüÿœ]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            docs.append(text)
            meta.append({"id": d.get("id"), "title": d.get("title"), "url": d.get("url")})
    return docs, meta

def build_index(docs):
    # Utilisation des stopwords français
    vectorizer = TfidfVectorizer(stop_words=french_stopwords, max_df=0.9, min_df=2, ngram_range=(1,2))
    tfidf = vectorizer.fit_transform(docs)
    return vectorizer, tfidf

def save_index(vectorizer, tfidf, meta):
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "tfidf": tfidf}, f)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

def main():
    print(f"Lecture des documents depuis: {DOCS_DIR}")
    docs, meta = load_documents()
    print(f"➡️ {len(docs)} documents chargés.")
    if len(docs) < 10:
        print(" Moins de 10 docs — ajoutez davantage avant d’indexer pour un meilleur résultat.")
    vectorizer, tfidf = build_index(docs)
    save_index(vectorizer, tfidf, meta)
    print(f" Index TF-IDF sauvegardé: {INDEX_PATH}")
    print(f" Métadonnées sauvegardées: {META_PATH}")

if __name__ == "__main__":
    main()
