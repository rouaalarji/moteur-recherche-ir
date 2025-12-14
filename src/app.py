
"""
Interface Web Flask - Moteur de Recherche
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

# Charger l'index
print(" Chargement de l'index...")
with open(INDEX_PATH, "rb") as f:
    index_data = pickle.load(f)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

vectorizer = index_data["vectorizer"]
tfidf_matrix = index_data["tfidf"]

bm25 = None
tokenized_docs = None
if "bm25" in index_data and "tokenized_docs" in index_data:
    bm25 = index_data["bm25"]
    tokenized_docs = index_data["tokenized_docs"]
    print(" BM25 chargé")
else:
    print(" BM25 non disponible")

print(f" {len(meta)} documents, {len(vectorizer.get_feature_names_out())} termes\n")

# Fonctions
def preprocess_query(query):
    query = query.lower()
    query = re.sub(r"[^\w\sàâçéèêëîïôûùüÿœ]", " ", query)
    return re.sub(r"\s+", " ", query).strip()

def tokenize(text):
    return preprocess_query(text).split()

def highlight_keywords(text, query):
    words = set(query.split())
    for word in words:
        if len(word) > 2:
            pattern = re.compile(f"(?i)({re.escape(word)})", re.IGNORECASE)
            text = pattern.sub(r"<mark>\1</mark>", text)
    return text

def get_excerpt(doc_id, query, max_length=300):
    doc_path = os.path.join(DOCS_DIR, f"doc_{doc_id:03d}.json")
    if not os.path.exists(doc_path):
        return ""
    try:
        with open(doc_path, encoding="utf-8") as f:
            content = json.load(f)
        text = content.get("content", "")
        if not text.strip():
            return ""
        
        text_lower = text.lower()
        query_words = query.split()
        best_pos = 0
        
        for word in query_words:
            pos = text_lower.find(word.lower())
            if pos != -1:
                best_pos = max(0, pos - max_length // 2)
                break
        
        excerpt = text[best_pos:best_pos + max_length]
        if best_pos > 0:
            excerpt = "..." + excerpt
        if best_pos + max_length < len(text):
            excerpt = excerpt + "..."
        
        return highlight_keywords(excerpt, query)
    except:
        return ""

def search_tfidf(query, page=1, per_page=5):
    start_time = time.time()
    clean_query = preprocess_query(query)
    if not clean_query:
        return [], 0, 0
    
    query_vec = vectorizer.transform([clean_query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    valid_indices = np.where(scores > 0)[0]
    valid_scores = scores[valid_indices]
    sorted_order = np.argsort(valid_scores)[::-1]
    ranked_indices = valid_indices[sorted_order]
    
    total_results = len(ranked_indices)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paged_indices = ranked_indices[start_idx:end_idx]
    
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
    if bm25 is None:
        return [], 0, 0
    
    start_time = time.time()
    q_tokens = tokenize(query)
    if not q_tokens:
        return [], 0, 0
    
    scores = bm25.get_scores(q_tokens)
    valid_indices = np.where(scores > 0)[0]
    valid_scores = scores[valid_indices]
    sorted_order = np.argsort(valid_scores)[::-1]
    ranked_indices = valid_indices[sorted_order]
    
    total_results = len(ranked_indices)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paged_indices = ranked_indices[start_idx:end_idx]
    
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
    return {
        "num_docs": len(meta),
        "num_terms": len(vectorizer.get_feature_names_out()),
        "bm25_available": bm25 is not None
    }

# Template HTML avec css
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Moteur de Recherche</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea, #764ba2);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px; margin: 0 auto; background: white; border-radius: 15px;padding: 30px;  box-shadow: 0 10px 30px rgba(0,0,0,0.3); 
        }
        
        h1 {
            text-align: center;  color: #667eea; margin-bottom: 10px;        }
        
        .subtitle {
            text-align: center; color: #666;margin-bottom: 20px; 
        }
        
        .stats {
            background: #f0f4ff;   padding: 15px;border-radius: 10px; margin-bottom: 20px;  text-align: center;color: #667eea;
        }
        
        .search-box {
            background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;
        }
        
        .search-input { display: flex; gap: 10px; margin-bottom: 15px;
        }
        
        .search-input input { flex: 1; padding: 12px 20px; border: 2px solid #ddd; border-radius: 25px; font-size: 16px;
        }
        
        .search-input input:focus { outline: none;border-color: #667eea;
        }
        
        .search-input button { padding: 12px 30px; background: linear-gradient(135deg, #667eea, #764ba2); border: none; border-radius: 25px;color: white; font-weight: bold; cursor: pointer;
        }
        
        .model-selector { display: flex; gap: 20px; justify-content: center;
        }
        
        .model-selector label { cursor: pointer;
        }
        
        .result-header { color: #667eea; font-size: 1.2rem; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;
        }
        
        .search-time { background: #10b981;  color: white;  padding: 5px 15px;  border-radius: 15px;  font-size: 0.9rem;
        }
        
        .result-card { background: #f8f9fa;   padding: 20px;   margin-bottom: 15px;   border-radius: 10px;   border-left: 4px solid #667eea;transition: transform 0.2s;
        }
        
        .result-card:hover { transform: translateX(5px);  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .result-title {color: #333;font-size: 1.2rem;font-weight: bold;margin-bottom: 10px;
        }
        
        .result-excerpt { color: #555; line-height: 1.6; margin-bottom: 10px;
        }
        .result-excerpt mark { background: #fff59d; padding: 2px 4px;  border-radius: 3px;
        }
        .result-meta {
            display: flex;  justify-content: space-between;  align-items: center;  font-size: 14px;
        }
        .result-score {
            background: #10b981;  color: white;  padding: 5px 12px;  border-radius: 15px;  font-weight: bold;
        }
        .result-link {  color: #667eea;  text-decoration: none;  font-weight: bold;
        }
        .result-link:hover {
            text-decoration: underline;
        }
        .comparison {   display: grid;  grid-template-columns: 1fr 1fr;  gap: 20px;
        }
        .comparison-col {  background: #f8f9fa;  padding: 15px;  border-radius: 10px;
        }
        .comparison-col h3 { text-align: center; color: #667eea;   margin-bottom: 15px;   padding-bottom: 10px;  border-bottom: 2px solid #667eea;
        }
        .comparison-col.bm25 h3 {  color: #764ba2;  border-color: #764ba2;
        }
        .pagination {  display: flex; justify-content: center; gap: 10px; margin-top: 20px;
        }
        
        .pagination a { padding: 8px 15px; border: 2px solid #ddd; border-radius: 8px; color: #667eea; text-decoration: none;
            transition: all 0.2s;
        }
        
        .pagination a:hover, .pagination a.active {  background: #667eea;color: white;  border-color: #667eea;
        }
        
        .no-results { text-align: center; padding: 40px; color: #999;
        }
        
        @media (max-width: 768px) {  .comparison { grid-template-columns: 1fr; }
  .search-input { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Moteur de Recherche</h1>
        <p class="subtitle">TF-IDF & BM25</p>
        
        <div class="stats">
            nombre des documents :{{ stats.num_docs }} documents | 
            nombre des termes : {{ "{:,}".format(stats.num_terms) }} termes | 
            🧠 TF-IDF{% if stats.bm25_available %} + BM25{% endif %}
        </div>
        
        <form method="POST" class="search-box">
            <div class="search-input">
                <input type="text" name="query" placeholder="Entrez votre requête..." 
                       value="{{ query }}" required>
                <button type="submit">Rechercher</button>
            </div>
            
            <div class="model-selector">
                <label>
                    <input type="radio" name="model" value="tfidf" 
                           {% if model == 'tfidf' %}checked{% endif %}> TF-IDF
                </label>
                {% if stats.bm25_available %}
                <label>
                    <input type="radio" name="model" value="bm25" 
                           {% if model == 'bm25' %}checked{% endif %}> BM25
                </label>
                <label>
                    <input type="radio" name="model" value="both" 
                           {% if model == 'both' %}checked{% endif %}> Comparer
                </label>
                {% endif %}
            </div>
        </form>
        
        {% if query %}
            {% if model == 'both' %}
                <div class="result-header">
                    <span>Résultats pour "<strong>{{ query }}</strong>"</span>
                </div>
                
                <div class="comparison">
                    <div class="comparison-col tfidf">
                        <h3>TF-IDF</h3>
                        {% if results_tfidf %}
                            {% for r in results_tfidf %}
                            <div class="result-card">
                                <div class="result-title">{{ r.title }}</div>
                                <div class="result-excerpt">{{ r.excerpt|safe }}</div>
                                <div class="result-meta">
                                    <span class="result-score">{{ r.score }}</span>
                                    <a href="{{ r.url }}" target="_blank" class="result-link">Voir →</a>
                                </div>
                            </div>
                            {% endfor %}
                        {% else %}
                            <div class="no-results"><p>Aucun résultat</p></div>
                        {% endif %}
                    </div>
                    
                    <div class="comparison-col bm25">
                        <h3>BM25</h3>
                        {% if results_bm25 %}
                            {% for r in results_bm25 %}
                            <div class="result-card">
                                <div class="result-title">{{ r.title }}</div>
                                <div class="result-excerpt">{{ r.excerpt|safe }}</div>
                                <div class="result-meta">
                                    <span class="result-score" style="background: #764ba2;">{{ r.score }}</span>
                                    <a href="{{ r.url }}" target="_blank" class="result-link">Voir →</a>
                                </div>
                            </div>
                            {% endfor %}
                        {% else %}
                            <div class="no-results"><p>Aucun résultat</p></div>
                        {% endif %}
                    </div>
                </div>
            {% else %}
                <div class="result-header">
                    <span>Résultats pour "<strong>{{ query }}</strong>"</span>
                    {% if search_time %}
                    <span class="search-time">⚡ {{ search_time }}s</span>
                    {% endif %}
                </div>
                
                {% if results %}
                    {% for r in results %}
                    <div class="result-card">
                        <div class="result-title">{{ r.title }}</div>
                        <div class="result-excerpt">{{ r.excerpt|safe }}</div>
                        <div class="result-meta">
                            <span class="result-score">Score: {{ r.score }}</span>
                            <a href="{{ r.url }}" target="_blank" class="result-link">Voir la source →</a>
                        </div>
                    </div>
                    {% endfor %}
                    
                    {% if total_pages > 1 %}
                    <div class="pagination">
                        {% if page > 1 %}
                        <a href="?query={{ query }}&model={{ model }}&page={{ page-1 }}">← Précédent</a>
                        {% endif %}
                        
                        {% for p in range(1, total_pages+1) %}
                        <a href="?query={{ query }}&model={{ model }}&page={{ p }}" 
                           {% if p == page %}class="active"{% endif %}>{{ p }}</a>
                        {% endfor %}
                        
                        {% if page < total_pages %}
                        <a href="?query={{ query }}&model={{ model }}&page={{ page+1 }}">Suivant →</a>
                        {% endif %}
                    </div>
                    <div style="text-align: center; color: #666; margin-top: 15px;">
                        Page {{ page }} sur {{ total_pages }} • {{ total_results }} résultat(s)
                    </div>
                    {% endif %}
                {% else %}
                    <div class="no-results">
                        <h3>Aucun résultat trouvé</h3>
                        <p>Essayez d'autres mots-clés</p>
                    </div>
                {% endif %}
            {% endif %}
        {% else %}
            <div class="no-results">
                <h3>👋 Bienvenue !</h3>
                <p>Entrez une requête pour commencer</p>
                <p style="font-size: 12px; margin-top: 10px;">
                    💡 Essayez: "intelligence artificielle", "blockchain" ou "climat"
                </p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    query = ""
    model = "tfidf"
    results = []
    results_tfidf = []
    results_bm25 = []
    search_time = None
    total_results = 0
    page = 1
    per_page = 5
    
    stats = get_stats()
    
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        model = request.form.get("model", "tfidf")
        page = 1
    else:
        query = request.args.get("query", "").strip()
        model = request.args.get("model", "tfidf")
        page = int(request.args.get("page", 1))
    
    if query:
        if model == "both":
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
                results, search_time, total_results = search_tfidf(query, page, per_page)
                model = "tfidf"
        else:
            results, search_time, total_results = search_tfidf(query, page, per_page)
    
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

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Serveur Flask démarré")
    print("="*60)
    print(f"📚 Documents: {len(meta)}")
    print(f"🔤 Termes: {len(vectorizer.get_feature_names_out()):,}")
    print(f"🧠 Modèles: TF-IDF" + (" + BM25" if bm25 else ""))
    print(f"🌐 URL: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host="127.0.0.1", port=5000)