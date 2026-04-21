from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect, LangDetectException
from functools import lru_cache


# ── Table des paires de langues disponibles ───────────────────────────
LANG_PAIRS = {
    ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en',
    ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
    ('ar', 'en'): 'Helsinki-NLP/opus-mt-ar-en',
    ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
    ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
    ('it', 'en'): 'Helsinki-NLP/opus-mt-it-en',
    ('en', 'ar'): 'Helsinki-NLP/opus-mt-en-ar',
    ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
    ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de',
    ('en', 'it'): 'Helsinki-NLP/opus-mt-en-it',
}

LANG_NAMES = {
    'fr': 'Français', 'en': 'Anglais', 'ar': 'Arabe',
    'es': 'Espagnol', 'de': 'Allemand', 'it': 'Italien', 'pt': 'Portugais'
}


# ── Chargement des modeles  ─────────
@lru_cache(maxsize=6)
def load_translation_model(src: str, tgt: str):
    """Charge le tokenizer et le modele pour une paire de langues."""
    key = (src, tgt)
    if key not in LANG_PAIRS:
        return None, None
    model_name = LANG_PAIRS[key]
    print(f'Chargement modele traduction {src}->{tgt} : {model_name}')
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


# ── Detection de langue ───────────────────────────────────────────────
def detect_language(text: str) -> str:
    """
    Detecte la langue d'un texte.
    Retourne un code ISO 639-1 (ex: 'fr', 'en', 'ar').
    Par defaut : 'en' si la detection echoue.
    """
    if not text or not text.strip():
        return 'en'
    try:
        return detect(text)
    except LangDetectException:
        return 'en'


# ── Traduction simple ─────────────────────────────────────────────────
def _translate_direct(text: str, src: str, tgt: str) -> str:
    """Traduit directement si la paire existe."""
    tokenizer, model = load_translation_model(src, tgt)
    if model is None:
        return None
    inputs = tokenizer(text, return_tensors='pt', padding=True,
                       truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=512,
                             num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ── Traduction avec pivot anglais ─────────────────────────────────────
def translate(text: str, src: str, tgt: str) -> str:
    """
    Traduit un texte de src vers tgt.
    Si la paire directe n'existe pas, passe par l'anglais (pivot).

    Args:
        text (str): Texte a traduire.
        src  (str): Langue source (code ISO).
        tgt  (str): Langue cible (code ISO).

    Returns:
        str: Texte traduit, ou message d'erreur.
    """
    if not text or not text.strip():
        return ''

    # Meme langue source et cible : retourner le texte tel quel
    if src == tgt:
        return text

    try:
        # Tentative de traduction directe
        result = _translate_direct(text, src, tgt)
        if result:
            return result

        # Traduction via pivot anglais : src -> en -> tgt
        if src != 'en':
            en_text = _translate_direct(text, src, 'en')
            if en_text and tgt != 'en':
                final = _translate_direct(en_text, 'en', tgt)
                return final if final else en_text
            elif en_text:
                return en_text

        return f'Traduction {src}->{tgt} non disponible.'

    except Exception as e:
        return f'Erreur de traduction : {str(e)}'


# ── Informations utiles ───────────────────────────────────────────────
def get_supported_pairs() -> list:
    """Retourne la liste des paires supportees."""
    return list(LANG_PAIRS.keys())


def get_lang_name(code: str) -> str:
    """Retourne le nom lisible d'un code de langue."""
    return LANG_NAMES.get(code, code.upper())


# ── Test rapide ───────────────────────────────────────────────────────
if __name__ == '__main__':
    tests = [
        ('Je suis tres content de ce produit !', 'fr', 'en'),
        ('This is a very good product.', 'en', 'fr'),
        ('Este restaurante es excelente.', 'es', 'en'),
    ]
    for text, src, tgt in tests:
        detected = detect_language(text)
        result = translate(text, src, tgt)
        print(f'[{src}->{tgt}] Detecte: {detected}')
        print(f'  Original : {text}')
        print(f'  Traduit  : {result}')
        print()