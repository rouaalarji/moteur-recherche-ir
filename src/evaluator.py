"""
Module d'évaluation 
- Métriques de base: Précision, Rappel, F1
- Comparaison TF-IDF vs BM25
"""
import os
import csv
import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === Configuration des chemins ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")
META_PATH = os.path.join(DATA_DIR, "meta.pkl")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.csv")
RESULTS_CSV = os.path.join(DATA_DIR, "evaluation_results.csv")
COMPARISON_CSV = os.path.join(DATA_DIR, "model_comparison.csv")

# === Charger l'index ===
print(" Chargement de l'index...")

with open(INDEX_PATH, "rb") as f:
    index_data = pickle.load(f)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

# TF-IDF 
vectorizer_tfidf = index_data["vectorizer"]
tfidf_matrix = index_data["tfidf"]

# BM25 
bm25_available = "bm25" in index_data and "tokenized_docs" in index_data
if bm25_available:
    bm25 = index_data["bm25"]
    tokenized_docs = index_data["tokenized_docs"]
    print(" TF-IDF et BM25 chargés\n")
else:
    bm25 = None
    tokenized_docs = None
    print("✅ TF-IDF chargé")
    print("⚠️  BM25 non disponible (ré-indexez avec le nouvel indexer.py)\n")

# === Fonctions utilitaires ===

def preprocess_query(query: str) -> str:
    """Nettoie et normalise la requête"""
    query = query.lower()
    query = re.sub(r"[^\w\sàâçéèêëîïôûùüÿœ]", " ", query)
    return re.sub(r"\s+", " ", query).strip()

def tokenize(text: str) -> list:
    """Tokenise le texte"""
    return preprocess_query(text).split()

# === Fonctions de recherche ===

def search_tfidf(query: str, top_k: int = 10):
    """Recherche avec TF-IDF - Retourne liste d'IDs"""
    clean_query = preprocess_query(query)
    query_vec = vectorizer_tfidf.transform([clean_query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    ranked_indices = np.argsort(scores)[::-1]
    results = []
    for idx in ranked_indices:
        if scores[idx] > 0 and len(results) < top_k:
            results.append(meta[idx]["id"])
    
    return results

def search_bm25(query: str, top_k: int = 10):
    """Recherche avec BM25 - Retourne liste d'IDs"""
    if bm25 is None:
        return []
    
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)
    
    ranked_indices = np.argsort(scores)[::-1]
    results = []
    for idx in ranked_indices:
        if scores[idx] > 0 and len(results) < top_k:
            results.append(meta[idx]["id"])
    
    return results

# === Métriques de base ===

def precision_recall_f1(pred_ids, true_ids):
    """Calcule Précision, Rappel et F1-mesure"""
    pred_set = set(pred_ids)
    true_set = set(true_ids)
    
    tp = len(pred_set & true_set)  # True Positives
    fp = len(pred_set - true_set)  # False Positives
    fn = len(true_set - pred_set)  # False Negatives
    
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    
    return precision, recall, f1

# === Chargement ground truth ===

def load_ground_truth():
    """Charge les requêtes de test depuis ground_truth.csv"""
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"❌ Fichier introuvable: {GROUND_TRUTH_PATH}")
        return []

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8-sig") as f:
        sample = f.read(1024)
        f.seek(0)
        
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            dialect = csv.get_dialect("excel")

        reader = csv.DictReader(f, dialect=dialect)
        raw_headers = reader.fieldnames or []
        headers = [h.strip().lower() for h in raw_headers]

        def find_col(cands):
            for c in cands:
                if c in headers:
                    return c
            return None

        col_query = find_col(["query", "requete", "q", "question"])
        col_docs = find_col(["doc_ids", "docs", "docids", "ids"])
        
        if not col_query or not col_docs:
            print("❌ En-têtes invalides dans ground_truth.csv")
            return []

        key_query_real = raw_headers[headers.index(col_query)]
        key_docs_real = raw_headers[headers.index(col_docs)]

        queries = []
        for row in reader:
            q = (row.get(key_query_real) or "").strip()
            ids_str = (row.get(key_docs_real) or "").strip()
            # Support virgule ET point-virgule
            ids = [int(x) for part in re.split(r"[;,]", ids_str) 
                   for x in [part.strip()] if x.isdigit()]
            if q and ids:
                queries.append((q, ids))
        
        return queries

# === Évaluation principale ===

def main(top_k=10):
    """Évaluation avec comparaison de modèles"""
    
    # Charger ground truth
    gt = load_ground_truth()
    if not gt:
        print("\n⚠️  Aucun ground truth valide.")
        print("Créez data/ground_truth.csv avec colonnes: query, doc_ids")
        print("\nExemple:")
        print("query,doc_ids")
        print('intelligence artificielle,"1;7;14"')
        return

    print("="*80)
    print(" "*20 + "ÉVALUATION DES MODÈLES DE RECHERCHE")
    print("="*80)
    print(f"\nNombre de requêtes de test: {len(gt)}\n")

    # Stocker les résultats
    rows_tfidf = []
    rows_bm25 = []
    
    # En-tête
    print(f"{'Modèle':<10} | {'Requête':<35} | {'Précision':>10} | {'Rappel':>10} | {'F1':>10}")
    print("-" * 85)
    
    # Évaluer chaque requête
    for query, true_ids in gt:
        query_short = query[:33] if len(query) > 33 else query
        
        # === TF-IDF ===
        pred_ids_tfidf = search_tfidf(query, top_k=top_k)
        p_tfidf, r_tfidf, f1_tfidf = precision_recall_f1(pred_ids_tfidf, true_ids)
        
        rows_tfidf.append([query, p_tfidf, r_tfidf, f1_tfidf])
        print(f"{'TF-IDF':<10} | {query_short:<35} | {p_tfidf:10.3f} | {r_tfidf:10.3f} | {f1_tfidf:10.3f}")
        
        # === BM25 (si disponible) ===
        if bm25_available:
            pred_ids_bm25 = search_bm25(query, top_k=top_k)
            p_bm25, r_bm25, f1_bm25 = precision_recall_f1(pred_ids_bm25, true_ids)
            
            rows_bm25.append([query, p_bm25, r_bm25, f1_bm25])
            print(f"{'BM25':<10} | {query_short:<35} | {p_bm25:10.3f} | {r_bm25:10.3f} | {f1_bm25:10.3f}")
        
        print()
    
    # === Moyennes ===
    print("-" * 85)
    
    avg_p_tfidf = np.mean([r[1] for r in rows_tfidf])
    avg_r_tfidf = np.mean([r[2] for r in rows_tfidf])
    avg_f1_tfidf = np.mean([r[3] for r in rows_tfidf])
    
    print(f"{'TF-IDF':<10} | {'MOYENNE':<35} | {avg_p_tfidf:10.3f} | {avg_r_tfidf:10.3f} | {avg_f1_tfidf:10.3f}")
    
    if bm25_available and rows_bm25:
        avg_p_bm25 = np.mean([r[1] for r in rows_bm25])
        avg_r_bm25 = np.mean([r[2] for r in rows_bm25])
        avg_f1_bm25 = np.mean([r[3] for r in rows_bm25])
        
        print(f"{'BM25':<10} | {'MOYENNE':<35} | {avg_p_bm25:10.3f} | {avg_r_bm25:10.3f} | {avg_f1_bm25:10.3f}")
    
    print("="*80)
    
    # === Sauvegarder les résultats ===
    
    # CSV TF-IDF
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "precision", "recall", "f1"])
        writer.writerows(rows_tfidf)
        writer.writerow(["MOYENNE", avg_p_tfidf, avg_r_tfidf, avg_f1_tfidf])
    
    print(f"\n✅ Résultats TF-IDF: {RESULTS_CSV}")
    
    # CSV Comparaison (si BM25 disponible)
    if bm25_available and rows_bm25:
        with open(COMPARISON_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query", "model", "precision", "recall", "f1"])
            for row in rows_tfidf:
                writer.writerow([row[0], "TF-IDF"] + row[1:])
            for row in rows_bm25:
                writer.writerow([row[0], "BM25"] + row[1:])
        
        print(f"✅ Comparaison: {COMPARISON_CSV}")
    
    # === Résumé final ===
    print("\n" + "="*80)
    print(" "*30 + "RÉSUMÉ")
    print("="*80)
    print(f"\n{'Modèle':<15} | {'Précision':>12} | {'Rappel':>12} | {'F1-mesure':>12}")
    print("-" * 80)
    print(f"{'TF-IDF':<15} | {avg_p_tfidf:12.3f} | {avg_r_tfidf:12.3f} | {avg_f1_tfidf:12.3f}")
    
    if bm25_available and rows_bm25:
        print(f"{'BM25':<15} | {avg_p_bm25:12.3f} | {avg_r_bm25:12.3f} | {avg_f1_bm25:12.3f}")
        
        # Différences
        diff_p = avg_p_bm25 - avg_p_tfidf
        diff_r = avg_r_bm25 - avg_r_tfidf
        diff_f1 = avg_f1_bm25 - avg_f1_tfidf
        better_model = "BM25" if diff_f1 > 0 else "TF-IDF"
        
        print("-" * 80)
        print(f"{'Différence':<15} | {diff_p:+12.3f} | {diff_r:+12.3f} | {diff_f1:+12.3f}")
        print(f"{'% Amélioration':<15} | {diff_p*100:+11.1f}% | {diff_r*100:+11.1f}% | {diff_f1*100:+11.1f}%")
        print("-" * 80)
        print(f"\n🏆 Meilleur modèle: {better_model}")
        print(f"   Amélioration F1: {abs(diff_f1):.3f} ({abs(diff_f1)*100:.1f}%)")
    
    print("="*80)
    print("\n✅ Évaluation terminée avec succès!\n")

if __name__ == "__main__":
    main(top_k=10)