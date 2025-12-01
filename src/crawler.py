
import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import time

# Dossier où stocker les documents on va crees le dossier 
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "documents")
os.makedirs(DATA_DIR, exist_ok=True)

# Liste d'URLs à scraper
URLS = [
    "https://fr.wikipedia.org/wiki/Intelligence_artificielle",
    "https://fr.wikipedia.org/wiki/Machine_learning",
    "https://fr.wikipedia.org/wiki/Big_data",
    "https://fr.wikipedia.org/wiki/Algorithme",
    "https://fr.wikipedia.org/wiki/Moteur_de_recherche",
    "https://fr.wikipedia.org/wiki/Tf-idf",
    "https://fr.wikipedia.org/wiki/Apprentissage_profond",
    "https://fr.wikipedia.org/wiki/Index_inverse",
    "https://fr.wikipedia.org/wiki/Mod%C3%A8le_vectoriel_(recherche_d%27information)",
    "https://fr.wikipedia.org/wiki/Cloud_computing",
    "https://fr.wikipedia.org/wiki/Cybers%C3%A9curit%C3%A9",
    "https://fr.wikipedia.org/wiki/Blockchain",
    "https://fr.wikipedia.org/wiki/Internet_des_objets",
    "https://fr.wikipedia.org/wiki/R%C3%A9seau_neuronal_artificiel",
    "https://fr.wikipedia.org/wiki/Deep_learning",
    "https://fr.wikipedia.org/wiki/Physique_quantique",
    "https://fr.wikipedia.org/wiki/Chimie_organique",
    "https://fr.wikipedia.org/wiki/Biologie_cellulaire",
    "https://fr.wikipedia.org/wiki/G%C3%A9n%C3%A9tique",
    "https://fr.wikipedia.org/wiki/Coronavirus",
    "https://fr.wikipedia.org/wiki/Vaccin",
    "https://fr.wikipedia.org/wiki/Diab%C3%A8te",
    "https://fr.wikipedia.org/wiki/M%C3%A9decine",
    "https://fr.wikipedia.org/wiki/Neurosciences",
    "https://fr.wikipedia.org/wiki/Climat",
    "https://fr.wikipedia.org/wiki/R%C3%A9chauffement_climatique",
    "https://fr.wikipedia.org/wiki/%C3%89conomie",
    "https://fr.wikipedia.org/wiki/Finance",
    "https://fr.wikipedia.org/wiki/Commerce_%C3%A9lectronique",
    "https://fr.wikipedia.org/wiki/Marketing",
    "https://fr.wikipedia.org/wiki/Entrepreneuriat",
    "https://fr.wikipedia.org/wiki/Crise_%C3%A9conomique",
    "https://fr.wikipedia.org/wiki/Inflation",
    "https://fr.wikipedia.org/wiki/Ch%C3%B4mage",
    "https://fr.wikipedia.org/wiki/Histoire",
    "https://fr.wikipedia.org/wiki/Art",
    "https://fr.wikipedia.org/wiki/Litt%C3%A9rature",
    "https://fr.wikipedia.org/wiki/Musique",
    "https://fr.wikipedia.org/wiki/Cin%C3%A9ma",
    "https://fr.wikipedia.org/wiki/Peinture",
    "https://fr.wikipedia.org/wiki/Architecture",
    "https://fr.wikipedia.org/wiki/Philosophie",
    "https://fr.wikipedia.org/wiki/Football",
    "https://fr.wikipedia.org/wiki/Basket-ball",
    "https://fr.wikipedia.org/wiki/Tennis",
    "https://fr.wikipedia.org/wiki/Natation",
    "https://fr.wikipedia.org/wiki/Athl%C3%A9tisme",
    "https://fr.wikipedia.org/wiki/Cyclisme",
    "https://fr.wikipedia.org/wiki/Rugby",
    "https://fr.wikipedia.org/wiki/Jeux_olympiques",
    "https://fr.wikipedia.org/wiki/Astronomie",
    "https://fr.wikipedia.org/wiki/Espace",
    "https://fr.wikipedia.org/wiki/Robotique",
    "https://fr.wikipedia.org/wiki/%C3%89nergie_renouvelable",
    "https://fr.wikipedia.org/wiki/%C3%89ducation",
    "https://fr.wikipedia.org/wiki/Psychologie",
    "https://fr.wikipedia.org/wiki/Sociologie",
    "https://fr.wikipedia.org/wiki/Intelligence_économique",
    "https://fr.wikipedia.org/wiki/Apprentissage_automatique",
    "https://fr.wikipedia.org/wiki/Robotique_médicale",
    "https://fr.wikipedia.org/wiki/Système_multi-agents",
    "https://fr.wikipedia.org/wiki/Science_des_données",
    "https://fr.wikipedia.org/wiki/Bio-informatique",
    "https://fr.wikipedia.org/wiki/Superordinateur",
    "https://fr.wikipedia.org/wiki/Cloud_privé",
    "https://fr.wikipedia.org/wiki/Algorithme_génétique",
    "https://fr.wikipedia.org/wiki/Optimisation_mathématique",
    "https://fr.wikipedia.org/wiki/Graphe_(mathématiques)",
    "https://fr.wikipedia.org/wiki/Théorie_des_jeux",
    "https://fr.wikipedia.org/wiki/Complexité_algorithme",
    "https://fr.wikipedia.org/wiki/Statistiques",
    "https://fr.wikipedia.org/wiki/Probabilité",
    "https://fr.wikipedia.org/wiki/Physique_nucléaire",
    "https://fr.wikipedia.org/wiki/Astrophysique",
    "https://fr.wikipedia.org/wiki/Chimie_inorganique",
    "https://fr.wikipedia.org/wiki/Microbiologie",
    "https://fr.wikipedia.org/wiki/Physiologie",
    "https://fr.wikipedia.org/wiki/Immunologie",
    "https://fr.wikipedia.org/wiki/Intelligence_émotionnelle",
    "https://fr.wikipedia.org/wiki/Neuropsychologie",
    "https://fr.wikipedia.org/wiki/Économie_comportementale",
    "https://fr.wikipedia.org/wiki/Management",
    "https://fr.wikipedia.org/wiki/Entrepreneuriat_social",
    "https://fr.wikipedia.org/wiki/Droit_du_travail",
    "https://fr.wikipedia.org/wiki/Globalisation",
    "https://fr.wikipedia.org/wiki/Sociologie_du_travail",
    "https://fr.wikipedia.org/wiki/Histoire_contemporaine",
    "https://fr.wikipedia.org/wiki/Mythologie_grecque",
    "https://fr.wikipedia.org/wiki/Histoire_romaine",
    "https://fr.wikipedia.org/wiki/Peinture_renaissance",
    "https://fr.wikipedia.org/wiki/Littérature_française",
    "https://fr.wikipedia.org/wiki/Théâtre_classique",
    "https://fr.wikipedia.org/wiki/Photographie",
    "https://fr.wikipedia.org/wiki/Sculpture",
    "https://fr.wikipedia.org/wiki/Boxe",
    "https://fr.wikipedia.org/wiki/Volleyball",
    "https://fr.wikipedia.org/wiki/Sports_mécaniques"
]

# Ajouter un User-Agent pour éviter le 403 de forbidden lorsque en ne peux pas acceder a une page
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MiniIRBot/1.0; +https://github.com/rouaalarji/moteur-recherche-ir.git)",
    "Accept-Language": "fr-FR,fr;q=0.9"
}
# ce la focntion fetch_page permet de recuperer le contenu dune page web
def fetch_page(url: str) -> dict:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title = soup.find("h1").get_text(strip=True)
    paragraphs = soup.select("div#mw-content-text p")
    content = "\n".join(p.get_text(" ", strip=True) for p in paragraphs if p.get_text(strip=True))

    return {
        "title": title,
        "url": url,
        "date": datetime.utcnow().isoformat(),
        "content": content
    }
# enregistrer le document dans un fichbier json
def save_doc(doc: dict, doc_id: int):
    path = os.path.join(DATA_DIR, f"doc_{doc_id:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"id": doc_id, **doc}, f, ensure_ascii=False, indent=2)

def main():
    print(f" Dossier de sortie: {DATA_DIR}")
    count = 0
    for i, url in enumerate(URLS, start=1):
        try:
            doc = fetch_page(url)
            if len(doc["content"]) < 300:
# pour eviter de sauvgareder les pages unitiles
                print(f" Contenu trop court, ignoré: {url}")
                continue
            save_doc(doc, i)
            count += 1
            print(f" Sauvé: {doc['title']} (id={i})")
            time.sleep(2)  # Pause pour éviter le blocage
        except Exception as e:
            print(f" Erreur pour {url}: {e}")

    print(f"\n Terminé: {count} documents sauvegardés.")
    print("➜ Ajoute plus d'URLs dans URLS pour atteindre ≥ 50 documents.")

if __name__ == "__main__":
    main()
