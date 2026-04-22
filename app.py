import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title='SentiTrad NLP',
    page_icon='',
    layout='wide',
    initial_sidebar_state='expanded'
)

try:
    from sentiment import analyze_sentiment
    from translation import translate, detect_language
    from utils import init_history, add_to_history
    MODULES_OK = True
except ImportError:
    MODULES_OK = False

if 'history' not in st.session_state:
    st.session_state.history = []

MAX_WORDS = 512

if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

def limit_text_words():
    words = st.session_state.text_input.split()
    if len(words) > MAX_WORDS:
        st.session_state.text_input = " ".join(words[:MAX_WORDS])

def clear_text():
    st.session_state.text_input = ""

st.markdown('''
<style>
    .metric-card {
        background: #f0f7ff;
        border-left: 4px solid #1A3C6E;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .title-main { color: #1A3C6E; font-size: 2.5rem; font-weight: bold; }
</style>
''', unsafe_allow_html=True)

with st.sidebar:
    # st.image('https://huggingface.co/front/assets/huggingface_logo.svg', width=120)
    st.title(' Paramètres')
    st.divider()

    target_lang = st.selectbox(
        ' Langue cible pour la traduction',
        options=['en', 'fr', 'es', 'de', 'ar', 'it', 'pt'],
        format_func=lambda x: {
            'en': '🇬🇧 Anglais', 'fr': '🇫🇷 Français',
            'es': '🇪🇸 Espagnol', 'de': '🇩🇪 Allemand',
            'ar': '🇸🇦 Arabe', 'it': '🇮🇹 Italien',
            'pt': '🇵🇹 Portugais'
        }.get(x, x)
    )

    show_all_scores = st.checkbox(' Afficher tous les scores', value=True)
    show_confidence = st.checkbox(' Afficher la confiance', value=True)
    st.divider()

    if st.session_state.history:
        st.subheader(' Historique')
        for item in reversed(st.session_state.history[-5:]):
            emoji = item.get('sentiment', {}).get('emoji', '●')
            texte = item.get('text', '')[:45]
            st.caption(f'{emoji} {texte}...')
    else:
        st.info('Aucune analyse encore.')

st.markdown('<p class="title-main"> SentiTrad NLP</p>', unsafe_allow_html=True)
st.caption('Analyse de sentiment & Traduction automatique par Transformers — HuggingFace')
st.divider()

text_input = st.text_area(
    ' Entrez votre texte ici (max 512 mots)',
    height=160,
    placeholder='Exemple : Ce restaurant est vraiment excellent, service impeccable !',
    key='text_input',
    on_change=limit_text_words
)

words = st.session_state.text_input.split()
if len(words) >= MAX_WORDS:
    st.warning(f" Texte limité à {MAX_WORDS} mots maximum.")

col_btn1, col_btn2, _ = st.columns([2, 2, 6])
with col_btn1:
    analyze_btn = st.button(' Analyser & Traduire', type='primary', use_container_width=True)
with col_btn2:
    clear_btn = st.button(' Effacer', use_container_width=True, on_click=clear_text)

if analyze_btn and text_input.strip():

    if not MODULES_OK:
        st.error(' Les modules sentiment.py et translation.py ne sont pas encore disponibles.')
        st.info("En attendant, voici un aperçu de l'interface.")
        demo_result = {
            'label': 'positive',
            'emoji': '😊',
            'scores': {'positive': 0.82, 'neutral': 0.12, 'negative': 0.06},
            'confidence': 82.0
        }
        demo_translation = '[Traduction indisponible - module non chargé]'
    else:
        demo_result = None
        demo_translation = None

    col1, col2 = st.columns(2, gap='large')

    with col1:
        st.subheader(' Analyse de Sentiment')
        with st.spinner('Analyse du sentiment en cours...'):
            if MODULES_OK:
                result = analyze_sentiment(text_input)
            else:
                result = demo_result

        if 'error' in result:
            st.error(result['error'])
        else:
            label_fr = {'positive': 'Positif', 'negative': 'Négatif', 'neutral': 'Neutre'}.get(result['label'], result['label'])
            color_map = {'positive': 'normal', 'negative': 'inverse', 'neutral': 'off'}
            st.metric(
                label='Sentiment détecté',
                value=f"{result['emoji']} {label_fr}",
                delta=f"{result['confidence']}% de confiance" if show_confidence else None,
                delta_color=color_map.get(result['label'], 'off')
            )

            if show_all_scores and 'scores' in result:
                labels = ['Négatif', 'Neutre', 'Positif']
                values = [
                    result['scores'].get('negative', 0),
                    result['scores'].get('neutral', 0),
                    result['scores'].get('positive', 0)
                ]
                colors = ['#e74c3c', '#95a5a6', '#2ecc71']
                fig = go.Figure(go.Bar(
                    x=values, y=labels, orientation='h',
                    marker_color=colors,
                    text=[f'{v:.1f}%' for v in values],
                    textposition='outside'
                ))
                fig.update_layout(
                    height=220, margin=dict(l=10, r=60, t=10, b=10),
                    xaxis=dict(range=[0, 110], showgrid=False),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(' Traduction Automatique')
        with st.spinner('Traduction en cours...'):
            if MODULES_OK:
                src_lang = detect_language(text_input)
                translated = translate(text_input, src_lang, target_lang)
            else:
                src_lang = 'fr'
                translated = demo_translation

        st.caption(f'Langue source détectée : **{src_lang.upper()}** → **{target_lang.upper()}**')
        st.text_area(' Texte traduit', value=translated, height=160, disabled=True)
        if MODULES_OK:
            st.download_button(' Télécharger la traduction', translated, 'traduction.txt')

    st.session_state.history.append({
        'text': text_input,
        'sentiment': result,
        'translation': translated if MODULES_OK else demo_translation
    })

elif analyze_btn and not text_input.strip():
    st.warning(" Veuillez entrer du texte avant d'analyser.")

with st.expander(' À propos de SentiTrad NLP'):
    st.markdown('''
    **SentiTrad NLP** est une application développée dans le cadre du module Applications IA.

    **Modèles utilisés :**
    - Sentiment : `cardiffnlp/twitter-roberta-base-sentiment-latest`
    - Traduction : `Helsinki-NLP/opus-mt-*` (MarianMT)
    - Détection de langue : `papluca/xlm-roberta-base-language-detection`

    **Équipe :** Salma TAMMARI - Wissal MAHBOUB - Hiba HAMDOUNI - Assmaa EL HIDANI
    ''')