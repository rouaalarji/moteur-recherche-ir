
 Moteur de Recherche IR
Un mini-moteur de recherche en Python basé sur le modèle vectoriel (TF-IDF).

## Installation
```bash
git clone <repo-url>
cd moteur-recherche-ir
pip install -r requirements.txt
Installer les dépendances :
Dans requirements.txt :
requests
beautifulsoup4
nltk
scikit-learn
flask   

# Mini Moteur de Recherche IR

## Description
Ce projet implémente un moteur de recherche basé sur le modèle vectoriel (TF-IDF) en Python.

## Fonctionnalités
- Collecte de documents (crawler)
- Prétraitement et indexation (TF-IDF)
- Recherche et ranking
- Interface CLI
- Évaluation (Précision, Rappel, F-mesure)

## Installation
```bash
git clone <URL>
cd moteur-recherche-ir
pip install -r requirements.txt

## 📥 Module de Collecte (crawler.py)

Ce module permet de **constituer le corpus de documents** pour le moteur de recherche.

### Rôle :
- Télécharger des pages web (ex. Wikipédia)
- Extraire le texte principal
- Sauvegarder chaque document en **JSON** avec :
  - id
  - titre
  - URL
  - date
  - contenu

### Utilisation :
```bash
python src/crawler.py
