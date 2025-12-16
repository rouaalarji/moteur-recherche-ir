
 #Mini-Moteur de Recherche Web en Python

Titre : Mini-moteur de recherche web en Python
 Développement d’un moteur de recherche permettant d’indexer et de rechercher des documents textuels en appliquant un modèle de Recherche d’Information (RI).

Objectifs :
Collecte et stockage de documents
Prétraitement et indexation
Recherche et ranking
Évaluation de la performance du moteur

2. Installation

-Cloner le dépôt :
git clone https://github.com/rouaalarji/moteur-recherche-ir.gitt
-Installer les dépendances :
pip install -r requirements.txt
3. Modèles de Recherche d’Information choisis + justification
  -TF-IDF (Term Frequency – Inverse Document Frequency)
  TF-IDF mesure l’importance d’un terme dans un document par rapport à l’ensemble du corpus.
Justification :
Simple à mettre en œuvre
Très utilisé dans les moteurs classiques
Basé sur une représentation vectorielle efficace

 -BM25 (Okapi BM25)
BM25 est un modèle probabiliste moderne, considéré comme l’un des meilleurs pour la recherche textuelle.
Justification :
Plus performant que TF-IDF dans la majorité des cas
Prend en compte :
la longueur des documents
la fréquence des termes
une saturation contrôlée des scores
3. Utilisation
A. Collecte de documents
Script : crawler.py
e crawler est le module qui s’occupe de collecter automatiquement les documents depuis le web pour constituer le corpus du moteur de recherche.
Objectif
  -Récupérer des pages web (articles Wikipedia dans ton cas).
  -Extraire uniquement le contenu utile (texte principal).
  -Sauvegarder chaque document avec ses métadonnées : titre, URL, date de récupération et contenu.
  -Créer un minimum de 50 documents pour le projet.
B-Script:indexer.py
Le module indexer.py a pour rôle de préparer les documents pour la recherche en créant les index nécessaires pour les modèles TF-IDF et BM25.
Objectif
  -Nettoyer et normaliser le texte des documents.
  -Tokeniser le texte et filtrer les stopwords français.
  -Construire les index pour deux modèles de Recherche d’Information :
  -TF-IDF : pour mesurer la pertinence des documents par rapport aux termes de la requête.
  -BM25 : pour un score probabiliste basé sur la fréquence des termes.
  -Sauvegarder les index et les métadonnées pour les utiliser dans l’interface et l’évaluation.
Fonctionnement général
Chargement des documents
Tous les fichiers JSON dans data/documents/ sont chargés.
Chaque document est nettoyé et normalisé (lowercase, suppression ponctuation et caractères spéciaux).
Les métadonnées (id, titre, URL) sont extraites et sauvegardées.
Normalisation et tokenisation
normalize_text(text) : convertit en minuscules, supprime ponctuation et espaces superflus.
tokenize(text) : découpe le texte en mots et supprime les stopwords français.
Construction des index
TF-IDF
Utilise TfidfVectorizer de scikit-learn.
Paramètres :
Stopwords français
max_df=0.9 : ignore les termes trop fréquents
min_df=2 : ignore les termes trop rares
ngram_range=(1,2) : prend les unigrammes et bigrammes
Produit une matrice TF-IDF : lignes = documents, colonnes = termes.
BM25
Chaque document est tokenisé.
Utilise BM25Okapi de rank_bm25.
Permet de calculer la pertinence probabiliste pour les requêtes.
C-Script:search_engine.py
Le module search_engine.py sert à interroger l’index des documents et à retourner les résultats les plus pertinents pour une requête donnée.
1 Objectif
Fournir une interface de recherche pour le moteur.
Utiliser les index construits dans indexer.py (TF-IDF et BM25).
Retourner les documents les plus pertinents avec leur score, titre et URL.

2 Fonctionnement général
  -Chargement des index
  -load_index() charge :
  -vectorizer et tfidf_matrix pour le TF-IDF
  -bm25 et tokenized_docs pour BM25
  -meta contenant les métadonnées (id, titre, URL)
  -Nettoyage et tokenisation de la requête
  -normalize_text(query) : minuscules, suppression de ponctuation et caractères spéciaux
  -tokenize(query) : découpe en mots et suppression des stopwords
Recherche selon le modèle:
TF-IDF:
La requête est transformée en vecteur TF-IDF.
cosine_similarity est utilisée pour comparer la requête à tous les documents.
Les documents sont classés par score décroissant et les top_k résultats sont retournés.
BM25:
La requête est tokenisée.
bm25.get_scores(q_tokens) calcule un score probabiliste pour chaque document.
Les documents sont triés par score décroissant et les top_k résultats sont retournés.
Format des résultats
Chaque résultat contient :
title : titre du document
url : URL du document
score : score de pertinence
id : identifiant du document
D-Script:app.py
L’interface web permet d’interagir avec le moteur de recherche via un navigateur, offrant une expérience utilisateur moderne et conviviale.

1 Objectifs
  Permettre la recherche de documents via TF-IDF et BM25.
  Comparer les résultats côte à côte (mode “both”).
  Afficher les résultats paginés avec extraits et score.
  Montrer les statistiques du moteur : nombre de documents, termes, disponibilité de BM25.
  Design responsive et moderne avec Bootstrap et surlignage des mots-clés.

2 Fonctionnement général
 -Chargement des index
 -Au démarrage, Flask charge l’index TF-IDF et BM25 (si disponible) ainsi que les métadonnées.
 -Prétraitement de la requête
 -La requête est normalisée et tokenisée pour le calcul des scores.
 -Les mots-clés sont surlignés dans les extraits des documents.

 Recherche
 - TF-IDF : vecteur de la requête comparé à la matrice TF-IDF via cosine_similarity.
 - BM25 : score probabiliste calculé sur les tokens de la requête.
 -Comparaison : affichage côte à côte des top 5 documents TF-IDF et BM25.
 -Pagination : résultats affichés 5 par page (modifiable via per_page).
 E-Script:evaluator.py
 Il compare plusieurs modèles (TF-IDF et BM25) et mesure leur performance sur un ensemble de requêtes test (ground truth).
 Objectifs du module
Évaluer la pertinence des résultats retournés par le moteur de recherche.
Comparer deux modèles de scoring :
  -TF-IDF
  -BM25
  -Générer des métriques standard en Recherche d’Information.
  -Produire des visualisations et rapports détaillés pour l’analyse.
Métriques calculées:
Précision (P)	:Proportion de résultats corrects parmi ceux retournés
Rappel (R):	Proportion des documents pertinents retrouvés
F1-Score:	Moyenne harmonique entre précision et rappel

Structure du projet:
moteur-recherche-ir/
│
├── src/                       
│    crawler.py             # Collecte des documents
│    indexer.py             # Indexation TF-IDF & BM25
│    search_engine.py       # Moteur de recherche CLI
│    app.py                 # Interface web Flask
│    evaluator.py           # Évaluation des performances
│
├── data/                       # Données
│    documents/             # Documents collectés (JSON)
│    index.pkl              # Index TF-IDF & BM25
│    meta.pkl               # Métadonnées
│    ground_truth.csv       # Vérité terrain
│   
├── requirements.txt           # Dépendances Python
├── README.md                  # Ce fichier
└── LICENSE                    # Licence MIT
