#  SentiTrad NLP

Application web de traitement du langage naturel (NLP) combinant :
- Analyse de sentiment
- Traduction automatique
- Détection de langue

---

##  Description

SentiTrad NLP est une application développée avec **Python** et **Streamlit** qui permet à un utilisateur de :

- Analyser le sentiment d’un texte (positif / négatif / neutre)
- Traduire automatiquement le texte vers une langue cible
- Détecter la langue d’entrée automatiquement

---

##  Technologies

- HuggingFace Transformers  
- Streamlit  
- PyTorch  
- Plotly  

---

##  Installation

```bash
git clone https://github.com/TON-USERNAME/sentitrad-nlp.git
cd sentitrad-nlp
```
```bash
python -m venv venv
venv\Scripts\activate   # ou source venv/bin/activate
```
```bash
pip install -r requirements.txt
streamlit run app.py
```
## Fonctionnalités
- Analyse de sentiment (RoBERTa)
- Traduction automatique (MarianMT)
- Détection de langue (XLM-R)
- Visualisation des résultats
- Historique des analyses

## Qualité du code
- GitHub Actions (tests automatiques)
- Linting avec flake8
- Merge bloqué si erreur

## Équipe
- Salma TAMMARI
- Wissal MAHBOUB
- Assmaa EL HIDANI
- Hiba HAMDOUNI
