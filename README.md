#  SentiTrad NLP

Application web de traitement du langage naturel (NLP) développée dans le cadre du module Applications IA — Transformers & NLP. combinant :

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

- HuggingFace Transformers : Modèles RoBERTa et MarianMT 
- Streamlit  : Interface web interactive
- PyTorch  : Backend deep learning
- Plotly  : Visualisations graphiques
- langdetect : Détection automatique de langue

---
## Modèles utilisés

- Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest
- Traduction: Helsinki-NLP/opus-mt-* (MarianMT)
- Fine-tuning: RoBERTa adapté sur dataset français

---

##  Installation

```bash
git clone https://github.com/SalmaTAMMARI12/sentitrad-nlp.git
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
- Visualisation des scores de confiance avec des graphiques interactifs
- Historique des analyses

## Qualité du code
- GitHub Actions (tests automatiques)
- Linting avec flake8
- Merge bloqué si erreur

## Équipe
- Salma TAMMARI
- Wissal MAHBOUB
- Hiba HAMDOUNI
- Assmaa EL HIDANI

