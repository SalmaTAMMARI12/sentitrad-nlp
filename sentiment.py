import re
from functools import lru_cache
from transformers import pipeline


# --- Constantes ---
MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

SENTIMENT_EMOJIS = {
    'positive': '\U0001f60a',
    'negative': '\U0001f61e',
    'neutral':  '\U0001f610'
}

LABEL_FR = {
    'positive': 'Positif',
    'negative': 'Negatif',
    'neutral':  'Neutre'
}


@lru_cache(maxsize=1)
def load_sentiment_model():
    print(f'Chargement du modele {MODEL_NAME}...')
    model = pipeline(
        task='sentiment-analysis',
        model=MODEL_NAME,
        top_k=None,
        truncation=True,
        max_length=512
    )
    print('Modele charge avec succes !')
    return model


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:1000]


def analyze_sentiment(text: str) -> dict:
    if not text or not text.strip():
        return {'error': 'Le texte est vide.'}

    clean = preprocess_text(text)
    if not clean:
        return {'error': 'Texte non analysable.'}

    try:
        model = load_sentiment_model()
    except Exception as e:
        return {'error': f'Erreur modele : {str(e)}'}

    try:
        raw_results = model(clean)[0]
    except Exception as e:
        return {'error': f'Erreur analyse : {str(e)}'}

    scores = {}
    for item in raw_results:
        label = item['label'].lower()
        if label == 'label_0': label = 'negative'
        if label == 'label_1': label = 'neutral'
        if label == 'label_2': label = 'positive'
        scores[label] = round(item['score'] * 100, 1)

    dominant = max(scores, key=scores.get)

    return {
        'label':      dominant,
        'label_fr':   LABEL_FR.get(dominant, dominant),
        'emoji':      SENTIMENT_EMOJIS.get(dominant, ''),
        'confidence': scores[dominant],
        'scores':     scores
    }


if __name__ == '__main__':
    tests = [
        'Ce restaurant est vraiment excellent, je recommande !',
        'Service catastrophique, jamais je ne reviendrai.',
        'Le produit est correct, ni bon ni mauvais.',
        'Hada mzyan bzzaf, chokran !',
        'This is absolutely amazing!'
    ]
    for t in tests:
        r = analyze_sentiment(t)
        if 'error' in r:
            print(f'ERREUR: {r["error"]}')
        else:
            print(f'{r["emoji"]} [{r["label_fr"]:8s} {r["confidence"]:5.1f}%] : {t[:50]}')