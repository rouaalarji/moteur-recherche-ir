#!/usr/bin/env python3
"""
Interface Web Flask - Moteur de Recherche
- Support TF-IDF et BM25
- Comparaison côte à côte
- Pagination fonctionnelle
- Statistiques système
- Design moderne
"""
from flask import Flask, render_template_string, request
import pickle
import os
import re
import json
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")
META_PATH = os.path.join(DATA_DIR, "meta.pkl")
DOCS_DIR = os.path.join(DATA_DIR, "documents")

# Charger l'index et les métadonnées
print("🔄 Chargement de l'index...")
with open(INDEX_PATH, "rb") as f:
    index_data = pickle.load(f)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

vectorizer = index_data["vectorizer"]
tfidf_matrix = index_data["tfidf"]

# BM25 (si disponible dans l'index)
bm25 = None
tokenized_docs = None
if "bm25" in index_data and "tokenized_docs" in index_data:
    bm25 = index_data["bm25"]
    tokenized_docs = index_data["tokenized_docs"]
    print("✅ BM25 chargé depuis l'index")
else:
    print("⚠️  BM25 non disponible - Utilisez le nouvel indexer.py")

print(f"✅ Index chargé: {len(meta)} documents, {len(vectorizer.get_feature_names_out())} termes\n")

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

def preprocess_query(query):
    """Nettoie et normalise la requête"""
    query = query.lower()
    query = re.sub(r"[^\w\sàâçéèêëîïôûùüÿœ]", " ", query)
    return re.sub(r"\s+", " ", query).strip()

def tokenize(text):
    """Tokenize le texte"""
    return preprocess_query(text).split()

def highlight_keywords(text, query):
    """Surligne les mots-clés dans le texte"""
    words = set(query.split())
    for word in words:
        if len(word) > 2:  # Ignorer les mots trop courts
            pattern = re.compile(f"(?i)({re.escape(word)})", re.IGNORECASE)
            text = pattern.sub(r"<mark>\1</mark>", text)
    return text

def get_excerpt(doc_id, query, max_length=300):
    """Récupère un extrait pertinent du document"""
    doc_path = os.path.join(DOCS_DIR, f"doc_{doc_id:03d}.json")
    
    if not os.path.exists(doc_path):
        return ""
    
    try:
        with open(doc_path, encoding="utf-8") as f:
            content = json.load(f)
        
        text = content.get("content", "")
        if len(text.strip()) == 0:
            return ""
        
        # Extraire un extrait centré sur les mots-clés
        text_lower = text.lower()
        query_words = query.split()
        best_pos = 0
        
        for word in query_words:
            pos = text_lower.find(word.lower())
            if pos != -1:
                best_pos = max(0, pos - max_length // 2)
                break
        
        # Extraire l'extrait
        excerpt = text[best_pos:best_pos + max_length]
        
        # Ajouter des ellipses
        if best_pos > 0:
            excerpt = "..." + excerpt
        if best_pos + max_length < len(text):
            excerpt = excerpt + "..."
        
        # Surligner les mots-clés
        excerpt = highlight_keywords(excerpt, query)
        
        return excerpt
    
    except Exception as e:
        return f"Erreur: {e}"

# ==========================================
# FONCTIONS DE RECHERCHE
# ==========================================

def search_tfidf(query, page=1, per_page=5):
    """Recherche avec TF-IDF"""
    start_time = time.time()
    
    clean_query = preprocess_query(query)
    if not clean_query:
        return [], 0, 0
    
    query_vec = vectorizer.transform([clean_query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Filtrer les scores > 0
    valid_indices = np.where(scores > 0)[0]
    valid_scores = scores[valid_indices]
    
    # Trier par score décroissant
    sorted_order = np.argsort(valid_scores)[::-1]
    ranked_indices = valid_indices[sorted_order]
    
    # Pagination
    total_results = len(ranked_indices)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paged_indices = ranked_indices[start_idx:end_idx]
    
    # Construire les résultats
    results = []
    for idx in paged_indices:
        doc_id = meta[idx]["id"]
        excerpt = get_excerpt(doc_id, clean_query)
        
        results.append({
            "title": meta[idx]["title"],
            "url": meta[idx]["url"],
            "score": round(float(scores[idx]), 4),
            "excerpt": excerpt
        })
    
    search_time = round(time.time() - start_time, 4)
    return results, search_time, total_results

def search_bm25(query, page=1, per_page=5):
    """Recherche avec BM25"""
    if bm25 is None:
        return [], 0, 0
    
    start_time = time.time()
    
    q_tokens = tokenize(query)
    if not q_tokens:
        return [], 0, 0
    
    scores = bm25.get_scores(q_tokens)
    
    # Filtrer les scores > 0
    valid_indices = np.where(scores > 0)[0]
    valid_scores = scores[valid_indices]
    
    # Trier par score décroissant
    sorted_order = np.argsort(valid_scores)[::-1]
    ranked_indices = valid_indices[sorted_order]
    
    # Pagination
    total_results = len(ranked_indices)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paged_indices = ranked_indices[start_idx:end_idx]
    
    # Construire les résultats
    results = []
    for idx in paged_indices:
        doc_id = meta[idx]["id"]
        excerpt = get_excerpt(doc_id, " ".join(q_tokens))
        
        results.append({
            "title": meta[idx]["title"],
            "url": meta[idx]["url"],
            "score": round(float(scores[idx]), 4),
            "excerpt": excerpt
        })
    
    search_time = round(time.time() - start_time, 4)
    return results, search_time, total_results

def get_stats():
    """Récupère les statistiques du système"""
    return {
        "num_docs": len(meta),
        "num_terms": len(vectorizer.get_feature_names_out()),
        "matrix_shape": tfidf_matrix.shape,
        "bm25_available": bm25 is not None
    }

# ==========================================
# TEMPLATE HTML
# ==========================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔍 Moteur de Recherche - TF-IDF & BM25</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .search-container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .search-header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .search-header h1 {
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .stats-bar {
            background: linear-gradient(135deg, #f0f4ff, #f8f0ff);
            padding: 15px 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            display: flex;
            justify-content: space-around;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .stat-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .stat-icon {
            color: #667eea;
            font-size: 1.2rem;
        }
        
        .stat-value {
            color: #667eea;
            font-weight: bold;
            font-size: 1.1rem;
        }
        
        .search-box {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        
        .search-input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .search-input-group input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 50px;
            font-size: 16px;
            transition: all 0.3s;
        }
        
        .search-input-group input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        }
        
        .search-input-group button {
            padding: 15px 40px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none;
            border-radius: 50px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
            white-space: nowrap;
        }
        
        .search-input-group button:hover {
            transform: scale(1.05);
        }
        
        .model-selector {
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .model-option {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .model-option input[type="radio"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
        
        .model-option label {
            cursor: pointer;
            font-weight: 500;
            color: #555;
            margin: 0;
        }
        
        .result-header {
            color: #667eea;
            font-size: 1.3rem;
            font-weight: bold;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .search-time {
            background: #e8f5e9;
            color: #2e7d32;
            padding: 8px 18px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: bold;
        }
        
        .result-card {
            background: linear-gradient(135deg, #f8f9fa, #ffffff);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            transition: all 0.3s;
            border: 2px solid transparent;
            position: relative;
        }
        
        .result-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 5px;
            height: 100%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .result-card:hover {
            transform: translateX(10px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border-color: #667eea;
        }
        
        .result-card:hover::before {
            opacity: 1;
        }
        
        .result-title {
            color: #667eea;
            font-size: 1.4rem;
            font-weight: bold;
            margin-bottom: 15px;
        }
        
        .result-excerpt {
            color: #555;
            line-height: 1.8;
            margin-bottom: 15px;
        }
        
        .result-excerpt mark {
            background: #fff59d;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: 600;
        }
        
        .result-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 15px;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .result-score {
            background: #10b981;
            color: white;
            padding: 6px 18px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
        }
        
        .result-link {
            color: #764ba2;
            text-decoration: none;
            font-weight: bold;
            transition: color 0.3s;
        }
        
        .result-link:hover {
            color: #667eea;
        }
        
        .comparison-view {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 20px;
        }
        
        .comparison-column {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
        }
        
        .comparison-column h3 {
            text-align: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid;
        }
        
        .comparison-column.tfidf h3 {
            color: #667eea;
            border-color: #667eea;
        }
        
        .comparison-column.bm25 h3 {
            color: #764ba2;
            border-color: #764ba2;
        }
        
        .pagination {
            margin-top: 30px;
        }
        
        .pagination .page-link {
            color: #667eea;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            margin: 0 5px;
            padding: 10px 15px;
            transition: all 0.3s;
        }
        
        .pagination .page-link:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        
        .pagination .page-item.active .page-link {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-color: #667eea;
        }
        
        .no-results {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }
        
        .no-results i {
            font-size: 4rem;
            color: #ddd;
            margin-bottom: 20px;
        }
        
        @media (max-width: 768px) {
            .comparison-view {
                grid-template-columns: 1fr;
            }
            
            .search-input-group {
                flex-direction: column;
            }
            
            .stats-bar {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="search-container">
        <!-- En-tête -->
        <div class="search-header">
            <h1><i class="fas fa-search"></i> Moteur de Recherche</h1>
            <p style="color: #666; font-size: 1.1rem;">TF-IDF & BM25 • Recherche intelligente</p>
        </div>
        
        <!-- Statistiques -->
        <div class="stats-bar">
            <div class="stat-item">
                <i class="fas fa-file-alt stat-icon"></i>
                <span>Documents: <span class="stat-value">{{ stats.num_docs }}</span></span>
            </div>
            <div class="stat-item">
                <i class="fas fa-font stat-icon"></i>
                <span>Termes: <span class="stat-value">{{ "{:,}".format(stats.num_terms) }}</span></span>
            </div>
            <div class="stat-item">
                <i class="fas fa-brain stat-icon"></i>
                <span>Modèles: <span class="stat-value">TF-IDF{% if stats.bm25_available %} + BM25{% endif %}</span></span>
            </div>
        </div>
        
        <!-- Formulaire de recherche -->
        <form method="POST" class="search-box">
            <div class="search-input-group">
                <input type="text" name="query" placeholder="Entrez votre requête..." 
                       value="{{ query }}" required autocomplete="off">
                <button type="submit">
                    <i class="fas fa-search"></i> Rechercher
                </button>
            </div>
            
            <div class="model-selector">
                <div class="model-option">
                    <input type="radio" id="tfidf" name="model" value="tfidf" 
                           {% if model == 'tfidf' %}checked{% endif %}>
                    <label for="tfidf">🎯 TF-IDF</label>
                </div>
                
                {% if stats.bm25_available %}
                <div class="model-option">
                    <input type="radio" id="bm25" name="model" value="bm25" 
                           {% if model == 'bm25' %}checked{% endif %}>
                    <label for="bm25">🚀 BM25</label>
                </div>
                
                <div class="model-option">
                    <input type="radio" id="both" name="model" value="both" 
                           {% if model == 'both' %}checked{% endif %}>
                    <label for="both">⚖️ Comparer les deux</label>
                </div>
                {% endif %}
            </div>
        </form>
        
        <!-- Résultats -->
        {% if query %}
            {% if model == 'both' %}
                <!-- Mode comparaison -->
                <div class="result-header">
                    <span>Résultats pour "<strong>{{ query }}</strong>"</span>
                </div>
                
                <div class="comparison-view">
                    <!-- Colonne TF-IDF -->
                    <div class="comparison-column tfidf">
                        <h3><i class="fas fa-chart-line"></i> TF-IDF</h3>
                        {% if results_tfidf %}
                            {% for r in results_tfidf %}
                            <div class="result-card">
                                <div class="result-title">{{ r.title }}</div>
                                <div class="result-excerpt">{{ r.excerpt|safe }}</div>
                                <div class="result-meta">
                                    <span class="result-score">Score: {{ r.score }}</span>
                                    <a href="{{ r.url }}" target="_blank" class="result-link">
                                        Voir <i class="fas fa-external-link-alt"></i>
                                    </a>
                                </div>
                            </div>
                            {% endfor %}
                        {% else %}
                            <div class="no-results">
                                <i class="fas fa-search-minus"></i>
                                <p>Aucun résultat</p>
                            </div>
                        {% endif %}
                    </div>
                    
                    <!-- Colonne BM25 -->
                    <div class="comparison-column bm25">
                        <h3><i class="fas fa-rocket"></i> BM25</h3>
                        {% if results_bm25 %}
                            {% for r in results_bm25 %}
                            <div class="result-card">
                                <div class="result-title">{{ r.title }}</div>
                                <div class="result-excerpt">{{ r.excerpt|safe }}</div>
                                <div class="result-meta">
                                    <span class="result-score" style="background: #764ba2;">Score: {{ r.score }}</span>
                                    <a href="{{ r.url }}" target="_blank" class="result-link">
                                        Voir <i class="fas fa-external-link-alt"></i>
                                    </a>
                                </div>
                            </div>
                            {% endfor %}
                        {% else %}
                            <div class="no-results">
                                <i class="fas fa-search-minus"></i>
                                <p>Aucun résultat</p>
                            </div>
                        {% endif %}
                    </div>
                </div>
            {% else %}
                <!-- Mode normal -->
                <div class="result-header">
                    <span>Résultats pour "<strong>{{ query }}</strong>"</span>
                    {% if search_time %}
                    <span class="search-time">
                        <i class="fas fa-bolt"></i> {{ search_time }}s
                    </span>
                    {% endif %}
                </div>
                
                {% if results %}
                    {% for r in results %}
                    <div class="result-card">
                        <div class="result-title">{{ r.title }}</div>
                        <div class="result-excerpt">{{ r.excerpt|safe }}</div>
                        <div class="result-meta">
                            <span class="result-score">Score: {{ r.score }}</span>
                            <a href="{{ r.url }}" target="_blank" class="result-link">
                                Voir la source <i class="fas fa-external-link-alt"></i>
                            </a>
                        </div>
                    </div>
                    {% endfor %}
                    
                    <!-- Pagination -->
                    {% if total_pages > 1 %}
                    <nav aria-label="Navigation">
                        <ul class="pagination justify-content-center">
                            {% if page > 1 %}
                            <li class="page-item">
                                <a class="page-link" href="?query={{ query }}&model={{ model }}&page={{ page-1 }}">
                                    <i class="fas fa-chevron-left"></i> Précédent
                                </a>
                            </li>
                            {% endif %}
                            
                            {% for p in range(1, total_pages+1) %}
                            <li class="page-item {% if p == page %}active{% endif %}">
                                <a class="page-link" href="?query={{ query }}&model={{ model }}&page={{ p }}">
                                    {{ p }}
                                </a>
                            </li>
                            {% endfor %}
                            
                            {% if page < total_pages %}
                            <li class="page-item">
                                <a class="page-link" href="?query={{ query }}&model={{ model }}&page={{ page+1 }}">
                                    Suivant <i class="fas fa-chevron-right"></i>
                                </a>
                            </li>
                            {% endif %}
                        </ul>
                    </nav>
                    
                    <div class="text-center" style="color: #666; margin-top: 15px;">
                        Page {{ page }} sur {{ total_pages }} • {{ total_results }} résultat(s) trouvé(s)
                    </div>
                    {% endif %}
                {% else %}
                    <div class="no-results">
                        <i class="fas fa-search-minus"></i>
                        <h3>Aucun résultat trouvé</h3>
                        <p>Essayez avec d'autres mots-clés ou termes plus généraux</p>
                    </div>
                {% endif %}
            {% endif %}
        {% else %}
            <div class="no-results">
                <i class="fas fa-hand-pointer"></i>
                <h3>Bienvenue sur le moteur de recherche !</h3>
                <p>Entrez une requête dans la barre de recherche pour commencer</p>
                <p style="color: #999; font-size: 0.9rem; margin-top: 20px;">
                    💡 Essayez: "intelligence artificielle", "machine learning" ou "climat"
                </p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

# ==========================================
# ROUTE PRINCIPALE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():
    """Route principale - Gère la recherche et l'affichage"""
    
    # Paramètres par défaut
    query = ""
    model = "tfidf"
    results = []
    results_tfidf = []
    results_bm25 = []
    search_time = None
    total_results = 0
    page = 1
    per_page = 5
    
    # Récupérer les statistiques
    stats = get_stats()
    
    # Gérer les paramètres
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        model = request.form.get("model", "tfidf")
        page = 1
    else:
        query = request.args.get("query", "").strip()
        model = request.args.get("model", "tfidf")
        page = int(request.args.get("page", 1))
    
    # Effectuer la recherche
    if query:
        if model == "both":
            # Mode comparaison : top 5 de chaque
            results_tfidf, _, _ = search_tfidf(query, page=1, per_page=5)
            results_bm25, _, _ = search_bm25(query, page=1, per_page=5)
            
            return render_template_string(
                HTML_TEMPLATE,
                query=query,
                model=model,
                results_tfidf=results_tfidf,
                results_bm25=results_bm25,
                stats=stats
            )
        
        elif model == "bm25":
            if bm25 is not None:
                results, search_time, total_results = search_bm25(query, page, per_page)
            else:
                # Fallback sur TF-IDF si BM25 non disponible
                results, search_time, total_results = search_tfidf(query, page, per_page)
                model = "tfidf"
        
        else:  # tfidf
            results, search_time, total_results = search_tfidf(query, page, per_page)
    
    # Calculer le nombre total de pages
    total_pages = (total_results + per_page - 1) // per_page if total_results > 0 else 0
    
    return render_template_string(
        HTML_TEMPLATE,
        query=query,
        model=model,
        results=results,
        search_time=search_time,
        total_results=total_results,
        page=page,
        total_pages=total_pages,
        stats=stats
    )

# ==========================================
# POINT D'ENTRÉE
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Serveur Flask démarré avec succès!")
    print("="*60)
    print(f"📊 Documents: {len(meta)}")
    print(f"📝 Termes: {len(vectorizer.get_feature_names_out()):,}")
    print(f"🧠 Modèles: TF-IDF" + (" + BM25" if bm25 else ""))
    print(f"🌐 URL: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host="127.0.0.1", port=5000)