#!/usr/bin/env python3
"""
Script pour générer automatiquement les statistiques du rapport
Lance ce script pour obtenir les VRAIES statistiques de ton système
"""
import os
import json
import pickle
from collections import Counter
import re

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DOCS_DIR = os.path.join(DATA_DIR, "documents")
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")
META_PATH = os.path.join(DATA_DIR, "meta.pkl")


def normalize_text(text: str) -> str:
    """Normalise le texte"""
    text = text.lower()
    text = re.sub(r"[^\w\sàâçéèêëîïôûùüÿœ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_corpus_statistics():
    """Génère les statistiques du corpus"""
    print("\n" + "="*80)
    print(" "*25 + " STATISTIQUES DU CORPUS")
    print("="*80 + "\n")
    
    # Charge les documents
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".json")]
    
    total_docs = 0
    total_words = 0
    all_tokens = []
    categories = Counter()
    
    print(" Analyse des documents...\n")
    
    for fname in files:
        fpath = os.path.join(DOCS_DIR, fname)
        
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                doc = json.load(f)
            
            content = doc.get("content", "")
            title = doc.get("title", "")
            
            # Compte les mots
            words = content.split()
            total_words += len(words)
            
            # Tokenise
            tokens = normalize_text(content).split()
            all_tokens.extend(tokens)
            
            # Catégorise (basé sur le titre)
            title_lower = title.lower()
            if any(k in title_lower for k in ["intelligence", "machine", "apprentissage", "algorithme", "données", "deep", "réseau"]):
                categories["IA et ML"] += 1
            elif any(k in title_lower for k in ["physique", "chimie", "biologie", "génétique"]):
                categories["Sciences"] += 1
            elif any(k in title_lower for k in ["médecine", "santé", "vaccin", "coronavirus", "diabète"]):
                categories["Médecine"] += 1
            elif any(k in title_lower for k in ["économie", "finance", "inflation", "commerce"]):
                categories["Économie"] += 1
            elif any(k in title_lower for k in ["football", "sport", "tennis", "basket", "athlé"]):
                categories["Sports"] += 1
            else:
                categories["Autres"] += 1
            
            total_docs += 1
            
        except Exception as e:
            print(f" Erreur: {fname} - {e}")
    
    # Calculs
    unique_tokens = len(set(all_tokens))
    avg_words = total_words / total_docs if total_docs > 0 else 0
    richness = (unique_tokens / len(all_tokens) * 100) if all_tokens else 0
    
    # Top termes
    token_counts = Counter(all_tokens)
    top_terms = token_counts.most_common(15)
    
    # Affiche les résultats
    print("="*80)
    print("STATISTIQUES GÉNÉRALES")
    print("="*80)
    print(f" Nombre total de documents       : {total_docs}")
    print(f" Nombre total de mots            : {total_words:,}")
    print(f" Moyenne de mots par document    : {avg_words:,.1f}")
    print(f" Nombre de tokens uniques        : {unique_tokens:,}")
    print(f" Richesse lexicale               : {richness:.2f}%")
    
    print("\n" + "="*80)
    print("RÉPARTITION THÉMATIQUE")
    print("="*80)
    for cat, count in categories.most_common():
        percentage = (count / total_docs * 100) if total_docs > 0 else 0
        print(f"{cat:<25} : {count:>3} documents ({percentage:>5.1f}%)")
    
    print("\n" + "="*80)
    print("TOP 15 TERMES LES PLUS FRÉQUENTS")
    print("="*80)
    for i, (term, count) in enumerate(top_terms, 1):
        print(f"{i:>2}. {term:<25} : {count:>6} occurrences")
    
    return {
        "total_docs": total_docs,
        "total_words": total_words,
        "avg_words": avg_words,
        "unique_tokens": unique_tokens,
        "richness": richness,
        "categories": dict(categories),
        "top_terms": top_terms
    }


def main():
    """Fonction principale"""
    print("\n" + "="*80)
    print(" "*20 + "📊 GÉNÉRATEUR DE STATISTIQUES")
    print("="*80)
    
    # Génère les statistiques
    corpus_stats = generate_corpus_statistics()


if __name__ == "__main__":
    main()