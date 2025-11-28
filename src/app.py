from flask import Flask, render_template_string, request
import pickle, os, re, json
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "..", "data", "index.pkl")
META_PATH = os.path.join(BASE_DIR, "..", "data", "meta.pkl")
DOCS_DIR = os.path.join(BASE_DIR, "..", "data", "documents")

# Charger index et métadonnées
with open(INDEX_PATH, "rb") as f:
    index_data = pickle.load(f)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

vectorizer = index_data["vectorizer"]
tfidf_matrix = index_data["tfidf"]

def preprocess_query(query):
    query = query.lower()
    query = re.sub(r"[^\w\sàâçéèêëîïôûùüÿœ]", " ", query)
    return re.sub(r"\s+", " ", query).strip()

def search(query, top_k=10):
    start_time = time.time()
    clean_query = preprocess_query(query)
    query_vec = vectorizer.transform([clean_query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in ranked_indices:
        if scores[idx] > 0:  # Ignorer les résultats avec score 0
            doc_path = os.path.join(DOCS_DIR, f"doc_{meta[idx]['id']:03d}.json")
            excerpt = ""
            if os.path.exists(doc_path):
                with open(doc_path, encoding="utf-8") as f:
                    content = json.load(f)
                text = content.get("content", "")
                excerpt = text[:300] + "..." if len(text) > 300 else text
            results.append({
                "title": meta[idx]["title"],
                "url": meta[idx]["url"],
                "score": round(float(scores[idx]), 4),
                "excerpt": excerpt
            })
    search_time = round(time.time() - start_time, 3)
    return results, search_time

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Moteur de Recherche - TF-IDF</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .search-container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            max-width: 1200px;
            margin: 0 auto;
        }
        .search-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .search-header h1 {
            color: #667eea;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
        }
        .search-box input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 50px;
            font-size: 16px;
            transition: all 0.3s;
        }
        .search-box input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        }
        .search-box button {
            padding: 15px 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 50px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .search-box button:hover {
            transform: scale(1.05);
        }
        .result-card {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            transition: all 0.3s;
            border: 2px solid transparent;
        }
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            border-color: #667eea;
        }
        .result-title {
            color: #667eea;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .result-excerpt {
            color: #666;
            line-height: 1.6;
            margin-bottom: 10px;
        }
        .result-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 15px;
        }
        .result-score {
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
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
        .no-results {
            text-align: center;
            padding: 40px;
            color: #999;
        }
        .stats {
            text-align: center;
            color: #666;
            margin-top: 20px;
            font-size: 14px;
        }
        .search-time {
            background: #e8f5e9;
            color: #2e7d32;
            padding: 10px 20px;
            border-radius: 10px;
            display: inline-block;
            margin-bottom: 20px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="search-container">
        <div class="search-header">
            <h1>🔍 Moteur de Recherche</h1>
            <p style="color: #666;">Recherche basée sur le modèle TF-IDF</p>
        </div>
        
        <form method="POST" class="search-box">
            <input type="text" name="query" placeholder="Entrez votre requête..." value="{{ query }}" required>
            <button type="submit">Rechercher</button>
        </form>
        
        {% if query %}
            <h4 style="color: #667eea; margin-bottom: 20px;">
                Résultats pour "<strong>{{ query }}</strong>"
            </h4>
            
            {% if search_time %}
            <div class="search-time">
                ⚡ Recherche effectuée en {{ search_time }} secondes
            </div>
            {% endif %}
            
            {% if results %}
                {% for r in results %}
                <div class="result-card">
                    <div class="result-title">{{ r.title }}</div>
                    <div class="result-excerpt">{{ r.excerpt }}</div>
                    <div class="result-meta">
                        <span class="result-score">Score: {{ r.score }}</span>
                        <a href="{{ r.url }}" target="_blank" class="result-link">Voir la source →</a>
                    </div>
                </div>
                {% endfor %}
                
                <div class="stats">
                    {{ results|length }} résultat(s) trouvé(s)
                </div>
            {% else %}
                <div class="no-results">
                    <h3>😕 Aucun résultat trouvé</h3>
                    <p>Essayez avec d'autres mots-clés</p>
                </div>
            {% endif %}
        {% else %}
            <div class="no-results">
                <h3>👋 Bienvenue !</h3>
                <p>Entrez une requête pour commencer la recherche</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    query = ""
    results = []
    search_time = None
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            results, search_time = search(query)
    return render_template_string(HTML_TEMPLATE, query=query, results=results, search_time=search_time)

if __name__ == "__main__":
    print("🚀 Serveur démarré sur http://127.0.0.1:5000")
    app.run(debug=True)