#!/usr/bin/env python3
"""
Module d'évaluation complet avec comparaison de modèles
- TF-IDF vs BM25
- Métriques: P, R, F1, MAP, MRR
- Courbes Précision-Rappel interpolées
- Rapport détaillé
"""
import os
import csv
import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Pour éviter les problèmes d'affichage

# === Configuration des chemins ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")
META_PATH = os.path.join(DATA_DIR, "meta.pkl")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.csv")
RESULTS_CSV = os.path.join(DATA_DIR, "evaluation_results.csv")
COMPARISON_CSV = os.path.join(DATA_DIR, "model_comparison.csv")
PR_CURVE_PATH = os.path.join(DATA_DIR, "precision_recall_curve.png")
COMPARISON_CHART = os.path.join(DATA_DIR, "model_comparison_chart.png")

# === Charger l'index ===
print("🔄 Chargement de l'index...")

with open(INDEX_PATH, "rb") as f:
    index_data = pickle.load(f)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

# TF-IDF (toujours disponible)
vectorizer_tfidf = index_data["vectorizer"]
tfidf_matrix = index_data["tfidf"]

# BM25 (si disponible)
bm25_available = "bm25" in index_data and "tokenized_docs" in index_data
if bm25_available:
    bm25 = index_data["bm25"]
    tokenized_docs = index_data["tokenized_docs"]
    print("✅ TF-IDF et BM25 chargés\n")
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
    """Recherche avec TF-IDF - Retourne [(doc_id, score), ...]"""
    clean_query = preprocess_query(query)
    query_vec = vectorizer_tfidf.transform([clean_query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    ranked_indices = np.argsort(scores)[::-1]
    results = []
    for idx in ranked_indices:
        if scores[idx] > 0 and len(results) < top_k:
            results.append((meta[idx]["id"], float(scores[idx])))
    
    return results

def search_bm25(query: str, top_k: int = 10):
    """Recherche avec BM25 - Retourne [(doc_id, score), ...]"""
    if bm25 is None:
        return []
    
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)
    
    ranked_indices = np.argsort(scores)[::-1]
    results = []
    for idx in ranked_indices:
        if scores[idx] > 0 and len(results) < top_k:
            results.append((meta[idx]["id"], float(scores[idx])))
    
    return results

def extract_ids(results):
    """Extrait les IDs depuis les résultats [(id, score), ...]"""
    return [doc_id for doc_id, _ in results]

# === Métriques de base ===

def precision_recall_f1(pred_ids, true_ids):
    """Calcule Précision, Rappel et F1"""
    pred_set = set(pred_ids)
    true_set = set(true_ids)
    
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    
    return precision, recall, f1

# === Métriques bonus ===

def average_precision(pred_results, true_ids):
    """Calcule Average Precision (AP)"""
    true_set = set(true_ids)
    if not true_set:
        return 0.0
    
    score = 0.0
    hit_count = 0
    
    for i, (doc_id, _) in enumerate(pred_results, start=1):
        if doc_id in true_set:
            hit_count += 1
            score += hit_count / i
    
    return score / len(true_set)

def mean_reciprocal_rank(pred_results, true_ids):
    """Calcule Reciprocal Rank (RR)"""
    true_set = set(true_ids)
    
    for i, (doc_id, _) in enumerate(pred_results, start=1):
        if doc_id in true_set:
            return 1.0 / i
    
    return 0.0

def interpolated_precision_recall(pred_results, true_ids):
    """Courbe P-R interpolée (11 points)"""
    true_set = set(true_ids)
    num_relevant = len(true_set)
    
    if num_relevant == 0:
        return [(r / 10, 0.0) for r in range(11)]
    
    # Calculer P/R pour chaque position
    pr_points = []
    num_correct = 0
    
    for rank, (doc_id, _) in enumerate(pred_results, start=1):
        if doc_id in true_set:
            num_correct += 1
            precision = num_correct / rank
            recall = num_correct / num_relevant
            pr_points.append((recall, precision))
    
    # Interpolation sur 11 points
    interpolated = []
    recall_levels = [r / 10 for r in range(11)]
    
    for level in recall_levels:
        max_precision = 0.0
        for r, p in pr_points:
            if r >= level:
                max_precision = max(max_precision, p)
        interpolated.append((level, max_precision))
    
    return interpolated

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

# === Génération des visualisations ===

def plot_precision_recall_curves(all_pr_curves_tfidf, all_pr_curves_bm25, queries):
    """Génère les courbes P-R avec comparaison TF-IDF vs BM25"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Courbe 1 : TF-IDF
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(queries))))
    for idx, (pr_curve, query) in enumerate(zip(all_pr_curves_tfidf[:10], queries[:10])):
        recalls = [r for r, p in pr_curve]
        precisions = [p for r, p in pr_curve]
        ax1.plot(recalls, precisions, marker='o', label=query[:25], 
                color=colors[idx], linewidth=2, markersize=4, alpha=0.7)
    
    # Moyenne TF-IDF
    if len(all_pr_curves_tfidf) > 1:
        mean_precisions = []
        recall_levels = [r / 10 for r in range(11)]
        for level in recall_levels:
            precisions_at_level = []
            for pr_curve in all_pr_curves_tfidf:
                for r, p in pr_curve:
                    if abs(r - level) < 0.01:
                        precisions_at_level.append(p)
                        break
            if precisions_at_level:
                mean_precisions.append(np.mean(precisions_at_level))
            else:
                mean_precisions.append(0.0)
        
        ax1.plot(recall_levels, mean_precisions, 'k--', linewidth=3, 
                label='Moyenne TF-IDF', marker='s', markersize=8)
    
    ax1.set_xlabel('Rappel', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Précision', fontsize=12, fontweight='bold')
    ax1.set_title('Courbes Précision-Rappel - TF-IDF', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    
    # Courbe 2 : BM25 (si disponible)
    if bm25_available and all_pr_curves_bm25:
        for idx, (pr_curve, query) in enumerate(zip(all_pr_curves_bm25[:10], queries[:10])):
            recalls = [r for r, p in pr_curve]
            precisions = [p for r, p in pr_curve]
            ax2.plot(recalls, precisions, marker='o', label=query[:25], 
                    color=colors[idx], linewidth=2, markersize=4, alpha=0.7)
        
        # Moyenne BM25
        if len(all_pr_curves_bm25) > 1:
            mean_precisions_bm25 = []
            for level in recall_levels:
                precisions_at_level = []
                for pr_curve in all_pr_curves_bm25:
                    for r, p in pr_curve:
                        if abs(r - level) < 0.01:
                            precisions_at_level.append(p)
                            break
                if precisions_at_level:
                    mean_precisions_bm25.append(np.mean(precisions_at_level))
                else:
                    mean_precisions_bm25.append(0.0)
            
            ax2.plot(recall_levels, mean_precisions_bm25, 'r--', linewidth=3, 
                    label='Moyenne BM25', marker='D', markersize=8)
        
        ax2.set_xlabel('Rappel', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Précision', fontsize=12, fontweight='bold')
        ax2.set_title('Courbes Précision-Rappel - BM25', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)
    else:
        ax2.text(0.5, 0.5, 'BM25 non disponible\nRé-indexez avec le nouvel indexer.py', 
                ha='center', va='center', fontsize=12, color='red')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(PR_CURVE_PATH, dpi=300, bbox_inches='tight')
    print(f"📊 Courbes P-R sauvegardées: {PR_CURVE_PATH}")
    plt.close()

def plot_model_comparison(rows_tfidf, rows_bm25):
    """Graphique de comparaison TF-IDF vs BM25"""
    if not bm25_available or not rows_bm25:
        return
    
    queries = [row[0][:20] for row in rows_tfidf[:10]]
    f1_tfidf = [row[3] for row in rows_tfidf[:10]]
    f1_bm25 = [row[3] for row in rows_bm25[:10]]
    
    x = np.arange(len(queries))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, f1_tfidf, width, label='TF-IDF', color='#667eea')
    ax.bar(x + width/2, f1_bm25, width, label='BM25', color='#764ba2')
    
    ax.set_xlabel('Requêtes', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison F1-Score : TF-IDF vs BM25', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(queries, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(COMPARISON_CHART, dpi=300, bbox_inches='tight')
    print(f"📊 Comparaison sauvegardée: {COMPARISON_CHART}")
    plt.close()

# === Évaluation principale ===

def main(top_k=10):
    """Évaluation complète avec comparaison de modèles"""
    
    # Charger ground truth
    gt = load_ground_truth()
    if not gt:
        print("\n⚠️  Aucun ground truth valide.")
        print("Créez data/ground_truth.csv avec colonnes: query, doc_ids")
        return

    print("="*80)
    print(" "*20 + "ÉVALUATION DES MODÈLES DE RECHERCHE")
    print("="*80)
    print(f"\nNombre de requêtes de test: {len(gt)}\n")

    # Stocker les résultats
    rows_tfidf = []
    rows_bm25 = []
    all_pr_curves_tfidf = []
    all_pr_curves_bm25 = []
    
    # En-tête
    print(f"{'Modèle':<8} | {'Requête':<30} | {'P':>6} | {'R':>6} | {'F1':>6} | {'AP':>6} | {'RR':>6}")
    print("-" * 90)
    
    # Évaluer chaque requête
    for query, true_ids in gt:
        query_short = query[:28] if len(query) > 28 else query
        
        # === TF-IDF ===
        results_tfidf = search_tfidf(query, top_k=top_k)
        pred_ids_tfidf = extract_ids(results_tfidf)
        
        p_tfidf, r_tfidf, f1_tfidf = precision_recall_f1(pred_ids_tfidf, true_ids)
        ap_tfidf = average_precision(results_tfidf, true_ids)
        rr_tfidf = mean_reciprocal_rank(results_tfidf, true_ids)
        pr_curve_tfidf = interpolated_precision_recall(results_tfidf, true_ids)
        
        all_pr_curves_tfidf.append(pr_curve_tfidf)
        rows_tfidf.append([query, p_tfidf, r_tfidf, f1_tfidf, ap_tfidf, rr_tfidf])
        
        print(f"{'TF-IDF':<8} | {query_short:<30} | {p_tfidf:6.3f} | {r_tfidf:6.3f} | {f1_tfidf:6.3f} | {ap_tfidf:6.3f} | {rr_tfidf:6.3f}")
        
        # === BM25 (si disponible) ===
        if bm25_available:
            results_bm25 = search_bm25(query, top_k=top_k)
            pred_ids_bm25 = extract_ids(results_bm25)
            
            p_bm25, r_bm25, f1_bm25 = precision_recall_f1(pred_ids_bm25, true_ids)
            ap_bm25 = average_precision(results_bm25, true_ids)
            rr_bm25 = mean_reciprocal_rank(results_bm25, true_ids)
            pr_curve_bm25 = interpolated_precision_recall(results_bm25, true_ids)
            
            all_pr_curves_bm25.append(pr_curve_bm25)
            rows_bm25.append([query, p_bm25, r_bm25, f1_bm25, ap_bm25, rr_bm25])
            
            print(f"{'BM25':<8} | {query_short:<30} | {p_bm25:6.3f} | {r_bm25:6.3f} | {f1_bm25:6.3f} | {ap_bm25:6.3f} | {rr_bm25:6.3f}")
        
        print()
    
    # === Moyennes ===
    print("-" * 90)
    
    avg_p_tfidf = np.mean([r[1] for r in rows_tfidf])
    avg_r_tfidf = np.mean([r[2] for r in rows_tfidf])
    avg_f1_tfidf = np.mean([r[3] for r in rows_tfidf])
    avg_map_tfidf = np.mean([r[4] for r in rows_tfidf])
    avg_mrr_tfidf = np.mean([r[5] for r in rows_tfidf])
    
    print(f"{'TF-IDF':<8} | {'MOYENNE':<30} | {avg_p_tfidf:6.3f} | {avg_r_tfidf:6.3f} | {avg_f1_tfidf:6.3f} | {avg_map_tfidf:6.3f} | {avg_mrr_tfidf:6.3f}")
    
    if bm25_available and rows_bm25:
        avg_p_bm25 = np.mean([r[1] for r in rows_bm25])
        avg_r_bm25 = np.mean([r[2] for r in rows_bm25])
        avg_f1_bm25 = np.mean([r[3] for r in rows_bm25])
        avg_map_bm25 = np.mean([r[4] for r in rows_bm25])
        avg_mrr_bm25 = np.mean([r[5] for r in rows_bm25])
        
        print(f"{'BM25':<8} | {'MOYENNE':<30} | {avg_p_bm25:6.3f} | {avg_r_bm25:6.3f} | {avg_f1_bm25:6.3f} | {avg_map_bm25:6.3f} | {avg_mrr_bm25:6.3f}")
    
    print("="*80)
    
    # === Sauvegarder les résultats ===
    
    # CSV TF-IDF
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "precision", "recall", "f1", "MAP", "MRR"])
        writer.writerows(rows_tfidf)
        writer.writerow(["MOYENNE", avg_p_tfidf, avg_r_tfidf, avg_f1_tfidf, avg_map_tfidf, avg_mrr_tfidf])
    
    print(f"\n✅ Résultats TF-IDF: {RESULTS_CSV}")
    
    # CSV Comparaison (si BM25 disponible)
    if bm25_available and rows_bm25:
        with open(COMPARISON_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["query", "model", "precision", "recall", "f1", "MAP", "MRR"])
            for row in rows_tfidf:
                writer.writerow([row[0], "TF-IDF"] + row[1:])
            for row in rows_bm25:
                writer.writerow([row[0], "BM25"] + row[1:])
        
        print(f"✅ Comparaison: {COMPARISON_CSV}")
    
    # === Générer les visualisations ===
    try:
        plot_precision_recall_curves(all_pr_curves_tfidf, all_pr_curves_bm25, [q for q, _ in gt])
        
        if bm25_available and rows_bm25:
            plot_model_comparison(rows_tfidf, rows_bm25)
    except Exception as e:
        print(f"\n⚠️  Erreur génération graphiques: {e}")
    
    # === Résumé final ===
    print("\n" + "="*80)
    print(" "*30 + "RÉSUMÉ")
    print("="*80)
    print(f"\n{'Modèle':<10} | {'Précision':>10} | {'Rappel':>10} | {'F1':>10} | {'MAP':>10} | {'MRR':>10}")
    print("-" * 80)
    print(f"{'TF-IDF':<10} | {avg_p_tfidf:10.3f} | {avg_r_tfidf:10.3f} | {avg_f1_tfidf:10.3f} | {avg_map_tfidf:10.3f} | {avg_mrr_tfidf:10.3f}")
    
    if bm25_available and rows_bm25:
        print(f"{'BM25':<10} | {avg_p_bm25:10.3f} | {avg_r_bm25:10.3f} | {avg_f1_bm25:10.3f} | {avg_map_bm25:10.3f} | {avg_mrr_bm25:10.3f}")
        
        # Différences
        diff_f1 = avg_f1_bm25 - avg_f1_tfidf
        diff_map = avg_map_bm25 - avg_map_tfidf
        better_model = "BM25" if diff_f1 > 0 else "TF-IDF"
        
        print("-" * 80)
        print(f"\n🏆 Meilleur modèle: {better_model}")
        print(f"   Différence F1: {abs(diff_f1):.3f} ({'+' if diff_f1 > 0 else '-'}{abs(diff_f1)*100:.1f}%)")
        print(f"   Différence MAP: {abs(diff_map):.3f} ({'+' if diff_map > 0 else '-'}{abs(diff_map)*100:.1f}%)")
    
    print("="*80)
    print("\n✅ Évaluation terminée avec succès!\n")

if __name__ == "__main__":
    main(top_k=10)