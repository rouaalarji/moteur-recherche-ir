#!/usr/bin/env python3
"""
Module d'indexation avec support TF-IDF et BM25
Compatible avec app.py et evaluator.py
"""
import os
import json
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords

# Télécharger les stopwords si nécessaire
try:
    french_stopwords = set(stopwords.words('french'))
except LookupError:
    print("📥 Téléchargement des stopwords NLTK...")
    nltk.download('stopwords')
    french_stopwords = set(stopwords.words('french'))

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DOCS_DIR = os.path.join(DATA_DIR, "documents")
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")
META_PATH = os.path.join(DATA_DIR, "meta.pkl")

# ==========================================
# FONCTIONS
# ==========================================

def normalize_text(text: str) -> str:
    """Nettoie et normalise le texte"""
    text = text.lower()
    text = re.sub(r"[^\w\sàâçéèêëîïôûùüÿœ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    """Tokenise le texte (sans stopwords)"""
    tokens = normalize_text(text).split()
    tokens = [t for t in tokens if t not in french_stopwords]
    return tokens

def load_documents():
    """Charge tous les documents JSON"""
    docs = []
    meta = []
    
    files = sorted([f for f in os.listdir(DOCS_DIR) if f.endswith(".json")])
    
    print(f"📁 Chargement de {len(files)} documents...")
    
    for fname in files:
        fpath = os.path.join(DOCS_DIR, fname)
        
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
            
            content = d.get("content", "")
            if len(content.strip()) == 0:
                print(f"⚠️  Document vide ignoré: {fname}")
                continue
            
            # Nettoyage
            clean_text = normalize_text(content)
            docs.append(clean_text)
            
            meta.append({
                "id": d.get("id"),
                "title": d.get("title"),
                "url": d.get("url")
            })
            
        except Exception as e:
            print(f"❌ Erreur lecture {fname}: {e}")
    
    print(f"✅ {len(docs)} documents chargés\n")
    return docs, meta

def build_tfidf_index(docs):
    """Construit l'index TF-IDF"""
    print("🔨 Construction de l'index TF-IDF...")
    
    vectorizer = TfidfVectorizer(
        stop_words=list(french_stopwords),
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    print(f"✅ Index TF-IDF créé:")
    print(f"   • Termes uniques: {len(vectorizer.get_feature_names_out()):,}")
    print(f"   • Taille matrice: {tfidf_matrix.shape}\n")
    
    return vectorizer, tfidf_matrix

def build_bm25_index(docs):
    """Construit l'index BM25"""
    print("🚀 Construction de l'index BM25...")
    
    tokenized_docs = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)
    
    print(f"✅ Index BM25 créé:")
    print(f"   • Documents tokenisés: {len(tokenized_docs)}")
    print(f"   • Moyenne tokens/doc: {sum(len(d) for d in tokenized_docs) / len(tokenized_docs):.1f}\n")
    
    return bm25, tokenized_docs

def save_index(vectorizer, tfidf_matrix, bm25, tokenized_docs, meta):
    """Sauvegarde l'index complet"""
    print("💾 Sauvegarde de l'index...")
    
    # Sauvegarder l'index
    index_data = {
        "vectorizer": vectorizer,
        "tfidf": tfidf_matrix,
        "bm25": bm25,
        "tokenized_docs": tokenized_docs
    }
    
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)
    
    # Sauvegarder les métadonnées
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)
    
    print(f"✅ Index sauvegardé: {INDEX_PATH}")
    print(f"✅ Métadonnées sauvegardées: {META_PATH}\n")

# ==========================================
# PROGRAMME PRINCIPAL
# ==========================================

def main():
    """Fonction principale"""
    print("\n" + "="*60)
    print(" "*15 + "INDEXATION DES DOCUMENTS")
    print("="*60 + "\n")
    
    # Vérifier que le dossier documents existe
    if not os.path.exists(DOCS_DIR):
        print(f"❌ Dossier introuvable: {DOCS_DIR}")
        print("   Exécutez d'abord crawler.py pour collecter les documents")
        return
    
    # Charger les documents
    docs, meta = load_documents()
    
    if len(docs) < 10:
        print("⚠️  Moins de 10 documents trouvés")
        print("   Exécutez crawler.py pour collecter plus de documents")
    
    # Construire TF-IDF
    vectorizer, tfidf_matrix = build_tfidf_index(docs)
    
    # Construire BM25
    try:
        bm25, tokenized_docs = build_bm25_index(docs)
        print("✅ BM25 disponible dans l'interface web\n")
    except Exception as e:
        print(f"⚠️  Erreur BM25: {e}")
        print("   Installez: pip install rank-bm25")
        print("   L'index sera créé sans BM25\n")
        bm25 = None
        tokenized_docs = []
    
    # Sauvegarder
    save_index(vectorizer, tfidf_matrix, bm25, tokenized_docs, meta)
    
    
    print(f"📊 Documents indexés: {len(docs)}")
    print(f"📝 Termes uniques: {len(vectorizer.get_feature_names_out()):,}")
  
    

if __name__ == "__main__":
    main()