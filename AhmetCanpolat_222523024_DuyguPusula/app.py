import streamlit as st
import time

from src.predict_berturk import get_analyzer

st.set_page_config(
    page_title="DuyguPusula",
    page_icon="💬",
    layout="wide"
)

label_map = {
    "mutluluk": "Mutluluk",
    "uzuntu": "Üzüntü",
    "ofke": "Öfke",
    "korku": "Korku",
    "saskinlik": "Şaşkınlık",
    "Mutluluk": "Mutluluk",
    "Üzüntü": "Üzüntü",
    "Öfke": "Öfke",
    "Korku": "Korku",
    "Şaşkınlık": "Şaşkınlık"
}

emoji_map = {
    "mutluluk": "😊",
    "uzuntu": "😔",
    "ofke": "😡",
    "korku": "😨",
    "saskinlik": "😮",
    "Mutluluk": "😊",
    "Üzüntü": "😔",
    "Öfke": "😡",
    "Korku": "😨",
    "Şaşkınlık": "😮"
}

emotion_theme = {
    "mutluluk": {
        "accent": "#f59e0b",
        "soft": "#fff7ed",
        "border": "#fed7aa",
        "comment": "Bu metinde olumlu, rahatlatıcı ve memnuniyet içeren ifadeler baskın görünüyor."
    },
    "uzuntu": {
        "accent": "#3b82f6",
        "soft": "#eff6ff",
        "border": "#bfdbfe",
        "comment": "Bu metinde kırgınlık, ayrılık, kayıp veya içsel burukluk taşıyan ifadeler öne çıkıyor."
    },
    "ofke": {
        "accent": "#ef4444",
        "soft": "#fef2f2",
        "border": "#fecaca",
        "comment": "Bu metinde rahatsızlık, tepki, haksızlık hissi veya sinirlenme anlamı baskın görünüyor."
    },
    "korku": {
        "accent": "#8b5cf6",
        "soft": "#f5f3ff",
        "border": "#ddd6fe",
        "comment": "Bu metinde kaygı, tedirginlik, panik veya güvenlik endişesi taşıyan ifadeler dikkat çekiyor."
    },
    "saskinlik": {
        "accent": "#06b6d4",
        "soft": "#ecfeff",
        "border": "#a5f3fc",
        "comment": "Bu metinde beklenmeyen bir durum karşısında oluşan şaşkınlık ve hayret duygusu öne çıkıyor."
    }
}

examples = [
    ("😊", "Mutluluk", "Uzun süredir beklediğim haber bugün geldi ve sonunda emeklerimin karşılığını aldığımı hissettim."),
    ("😔", "Üzüntü", "Mezuniyet yaklaştıkça sevdiğim arkadaşlarımdan ayrılacak olmak içimde buruk bir his bırakıyor."),
    ("😡", "Öfke", "Bütün sorumluluğu bana bırakıp sonra beni suçlamaları gerçekten sinirimi bozdu."),
    ("😨", "Korku", "Gece yolda yürürken arkamdan gelen ayak sesleri yüzünden tedirgin olmaya başladım."),
    ("😮", "Şaşkınlık", "Hiç beklemediğim bir anda ismimin anons edilmesi beni birkaç saniye ne diyeceğimi bilemeyecek kadar şaşırttı.")
]

all_labels = ["mutluluk", "uzuntu", "ofke", "korku", "saskinlik"]

@st.cache_resource
def load_model():
    return get_analyzer()

def normalize_key(value):
    value = str(value).strip().lower()
    value = value.replace("ı", "i")
    value = value.replace("ğ", "g")
    value = value.replace("ü", "u")
    value = value.replace("ş", "s")
    value = value.replace("ö", "o")
    value = value.replace("ç", "c")
    return value

def normalize_label(label):
    key = normalize_key(label)
    return label_map.get(label, label_map.get(key, str(label)))

def get_emoji(label):
    key = normalize_key(label)
    return emoji_map.get(label, emoji_map.get(key, "💬"))

def get_theme(label):
    key = normalize_key(label)
    return emotion_theme.get(key, {
        "accent": "#2563eb",
        "soft": "#eff6ff",
        "border": "#bfdbfe",
        "comment": "Bu metinde modelin belirlediği duygu sınıfı baskın görünüyor."
    })

def to_percent(value):
    if value is None:
        return None

    try:
        value = str(value).replace("%", "").replace(",", ".")
        value = float(value)
    except:
        return None

    if value <= 1:
        value *= 100

    return value

def predict_text(analyzer, text):
    if hasattr(analyzer, "predict"):
        return analyzer.predict(text)
    if hasattr(analyzer, "analyze"):
        return analyzer.analyze(text)
    if hasattr(analyzer, "predict_emotion"):
        return analyzer.predict_emotion(text)
    if callable(analyzer):
        return analyzer(text)
    raise AttributeError("Tahmin metodu bulunamadı.")

def parse_scores(scores):
    parsed = {}

    if isinstance(scores, dict):
        for label, value in scores.items():
            key = normalize_key(label)
            percent = to_percent(value)

            if percent is not None:
                parsed[key] = percent

    if isinstance(scores, list):
        for item in scores:
            if isinstance(item, dict):
                label = item.get("label") or item.get("duygu") or item.get("emotion")
                value = item.get("score") or item.get("guven") or item.get("probability")

                if label is not None:
                    key = normalize_key(label)
                    percent = to_percent(value)

                    if percent is not None:
                        parsed[key] = percent

    return parsed

def parse_result(result):
    if not isinstance(result, dict):
        return {
            "main_label": str(result),
            "confidence": None,
            "second_label": None,
            "second_confidence": None,
            "scores": {}
        }

    main_label = (
        result.get("ana_duygu")
        or result.get("duygu")
        or result.get("label")
        or result.get("emotion")
        or result.get("prediction")
        or result.get("predicted_label")
        or result.get("main_emotion")
    )

    confidence = (
        result.get("guven")
        or result.get("confidence")
        or result.get("score")
        or result.get("probability")
        or result.get("ana_guven")
    )

    second_label = (
        result.get("ikinci_duygu")
        or result.get("ikinci_olasi_duygu")
        or result.get("second_label")
        or result.get("second_emotion")
    )

    second_confidence = (
        result.get("ikinci_guven")
        or result.get("ikinci_score")
        or result.get("second_score")
        or result.get("second_confidence")
    )

    raw_scores = (
        result.get("skorlar")
        or result.get("scores")
        or result.get("olasiliklar")
        or result.get("probabilities")
        or result.get("class_scores")
        or result.get("tum_skorlar")
        or {}
    )

    scores = parse_scores(raw_scores)

    if len(scores) > 0:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if main_label is None:
            main_label = sorted_scores[0][0]
            confidence = sorted_scores[0][1]

        if second_label is None and len(sorted_scores) > 1:
            second_label = sorted_scores[1][0]
            second_confidence = sorted_scores[1][1]

    main_key = normalize_key(main_label) if main_label is not None else None
    second_key = normalize_key(second_label) if second_label is not None else None

    main_percent = to_percent(confidence)
    second_percent = to_percent(second_confidence)

    if main_key is not None and main_percent is not None and main_key not in scores:
        scores[main_key] = main_percent

    if second_key is not None and second_percent is not None and second_key not in scores:
        scores[second_key] = second_percent

    return {
        "main_label": main_label,
        "confidence": confidence,
        "second_label": second_label,
        "second_confidence": second_confidence,
        "scores": scores
    }

def render_score_rows(scores):
    if scores is None or len(scores) == 0:
        return '<div class="score-empty">Duygu skorları bulunamadı.</div>'

    fixed_scores = {}

    for label in all_labels:
        fixed_scores[label] = scores.get(label, 0)

    sorted_scores = sorted(fixed_scores.items(), key=lambda x: x[1], reverse=True)

    html = '<div class="score-list">'

    for label, score in sorted_scores:
        display_label = normalize_label(label)
        emoji = get_emoji(label)
        width = max(2, min(score, 100))
        html += f'<div class="score-row"><div class="score-left"><span class="score-emoji">{emoji}</span><span class="score-label">{display_label}</span></div><div class="score-middle"><div class="score-track"><div class="score-fill" style="width:{width}%;"></div></div></div><div class="score-value">%{score:.2f}</div></div>'

    html += '</div>'
    return html

def render_result(result=None, elapsed=None):
    if result is None:
        html = '<div class="empty-result"><div class="empty-icon">💬</div><div class="empty-title">Sonuç Bekleniyor</div><div class="empty-text">Metni yazıp analiz et butonuna bastığında ana duygu, güven skoru ve tüm duygu oranları burada görünecek.</div></div>'
        st.markdown(html, unsafe_allow_html=True)
        return

    main_label = result["main_label"]
    confidence = to_percent(result["confidence"])
    scores = result["scores"]

    emotion = normalize_label(main_label)
    emoji = get_emoji(main_label)
    theme = get_theme(main_label)
    confidence_text = f"%{confidence:.2f}" if confidence is not None else "Bilinmiyor"
    score_rows = render_score_rows(scores)
    elapsed_line = f"Analiz süresi: {elapsed:.2f} saniye" if elapsed is not None else ""

    html = f'<div class="result-colored" style="border-color:{theme["border"]}; background:linear-gradient(180deg, #ffffff 0%, {theme["soft"]} 100%);"><div class="main-result" style="border-color:{theme["border"]};"><div class="emotion-emoji">{emoji}</div><div class="emotion-name" style="color:{theme["accent"]};">{emotion}</div><div class="confidence-box" style="color:{theme["accent"]}; border-color:{theme["border"]};">Güven Skoru: {confidence_text}</div></div><div class="emotion-comment">{theme["comment"]}</div><div class="score-title">Tüm Duygu Skorları</div>{score_rows}<div class="elapsed-line">{elapsed_line}</div></div>'
    st.markdown(html, unsafe_allow_html=True)

def use_example(sentence):
    st.session_state.input_text = sentence
    st.session_state.analysis_result = None
    st.session_state.analysis_time = None

st.markdown(
"""
<style>
.stApp {
    background: linear-gradient(180deg, #ffffff 0%, #f4f9ff 100%);
}

.block-container {
    max-width: 1120px;
    padding-top: 18px;
    padding-bottom: 18px;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.hero {
    display: flex;
    align-items: center;
    gap: 18px;
    padding-top: 8px;
}

.hero-icon {
    width: 76px;
    height: 76px;
    border-radius: 24px;
    background: linear-gradient(135deg, #eaf4ff, #ffffff);
    border: 1px solid #bfdbfe;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 36px;
    box-shadow: 0 14px 34px rgba(37, 99, 235, 0.15);
}

.hero-title {
    font-size: 44px;
    line-height: 1.02;
    font-weight: 950;
    color: #1556d6;
    letter-spacing: 0.2px;
    margin: 0;
}

.hero-title span {
    color: #0f2f63;
}

.hero-subtitle {
    color: #64748b;
    font-size: 16.5px;
    margin-top: 9px;
    line-height: 1.45;
}

.badge-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 13px;
}

.model-badge {
    background: #eff6ff;
    color: #1556d6;
    border: 1px solid #bfdbfe;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 12.5px;
    font-weight: 800;
}

.input-title {
    color: #1556d6;
    font-size: 18px;
    font-weight: 850;
    margin-bottom: 8px;
}

.panel-title {
    color: #1556d6;
    font-size: 20px;
    font-weight: 900;
    margin-bottom: 10px;
}

.panel-help {
    color: #64748b;
    font-size: 13px;
    margin-top: -4px;
    margin-bottom: 12px;
}

.result-colored {
    border: 1px solid #bfdbfe;
    border-radius: 16px;
    padding: 12px;
    min-height: auto;
}

.main-result {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding-top: 0px;
    padding-bottom: 8px;
    border-bottom: 1px solid #dbeafe;
}

.emotion-emoji {
    font-size: 52px;
    line-height: 1;
    margin-bottom: 4px;
}

.emotion-name {
    font-size: 30px;
    line-height: 1.05;
    font-weight: 950;
    margin-bottom: 8px;
}

.confidence-box {
    background: #ffffff;
    border: 1px solid #bfdbfe;
    border-radius: 999px;
    padding: 7px 16px;
    font-size: 14px;
    font-weight: 850;
}

.emotion-comment {
    color: #475569;
    font-size: 12.5px;
    line-height: 1.3;
    text-align: center;
    background: rgba(255,255,255,0.75);
    border: 1px solid #e1edff;
    border-radius: 12px;
    padding: 7px 10px;
    margin-top: 8px;
}

.score-title {
    color: #0f2f63;
    font-size: 14px;
    font-weight: 900;
    margin-top: 10px;
    margin-bottom: 7px;
}

.score-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.score-row {
    display: grid;
    grid-template-columns: 112px 1fr 58px;
    align-items: center;
    gap: 9px;
    background: #ffffff;
    border: 1px solid #e1edff;
    border-radius: 12px;
    padding: 6px 9px;
}

.score-left {
    display: flex;
    align-items: center;
    gap: 8px;
}

.score-emoji {
    font-size: 18px;
}

.score-label {
    color: #1e293b;
    font-size: 13px;
    font-weight: 750;
}

.score-track {
    width: 100%;
    height: 7px;
    background: #e7f0ff;
    border-radius: 999px;
    overflow: hidden;
}

.score-fill {
    height: 100%;
    background: linear-gradient(90deg, #1556d6, #60a5fa);
    border-radius: 999px;
}

.score-value {
    color: #1556d6;
    font-size: 12.5px;
    font-weight: 850;
    text-align: right;
}

.empty-result {
    min-height: 300px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.empty-icon {
    font-size: 58px;
    margin-bottom: 8px;
}

.empty-title {
    color: #1556d6;
    font-size: 26px;
    font-weight: 900;
    margin-bottom: 8px;
}

.empty-text {
    color: #64748b;
    font-size: 15px;
    text-align: center;
    width: 82%;
    line-height: 1.4;
}

.elapsed-line {
    color: #94a3b8;
    font-size: 12px;
    text-align: center;
    margin-top: 10px;
}

div[data-testid="stTextArea"] textarea {
    background: #ffffff;
    color: #0f172a;
    border: 1px solid #bfdbfe;
    border-radius: 15px;
    min-height: 102px !important;
    font-size: 15.5px;
    box-shadow: none;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #1556d6;
    box-shadow: 0 0 0 2px rgba(21, 86, 214, 0.12);
}

button[kind="primary"] {
    background: linear-gradient(90deg, #1556d6, #60a5fa) !important;
    color: white !important;
    border: none !important;
    border-radius: 15px !important;
    height: 44px !important;
    font-size: 15px !important;
    font-weight: 850 !important;
    box-shadow: 0 12px 24px rgba(37, 99, 235, 0.23) !important;
}

button[kind="secondary"] {
    background: #f8fbff !important;
    color: #0f2f63 !important;
    border: 1px solid #dbeafe !important;
    border-radius: 14px !important;
    min-height: 56px !important;
    font-size: 13.5px !important;
    text-align: left !important;
    white-space: normal !important;
    box-shadow: none !important;
}

button[kind="secondary"]:hover {
    border-color: #60a5fa !important;
    background: #eff6ff !important;
}

.footer-text {
    color: #94a3b8;
    text-align: center;
    font-size: 12px;
    margin-top: 10px;
}
</style>
""",
    unsafe_allow_html=True
)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_time" not in st.session_state:
    st.session_state.analysis_time = None

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

top_left, top_right = st.columns([0.9, 1.25], gap="large")

with top_left:
    st.markdown('<div class="hero"><div class="hero-icon">💬</div><div><div class="hero-title">Duygu<span>Pusula</span></div><div class="hero-subtitle">Türkçe metinler için BERTurk tabanlı duygu sınıflandırma sistemi</div><div class="badge-row"><div class="model-badge">BERTurk Modeli</div><div class="model-badge">5 Duygu Sınıfı</div><div class="model-badge">Test Başarısı: %97.63</div></div></div></div>', unsafe_allow_html=True)

with top_right:
    st.markdown('<div class="input-title">✏️ Metninizi Girin</div>', unsafe_allow_html=True)
    user_text = st.text_area("Metin", label_visibility="collapsed", placeholder="Analiz etmek istediğiniz metni buraya yazın...", key="input_text", height=102)
    analyze_clicked = st.button("✨ Analiz Et", use_container_width=True, type="primary")

if analyze_clicked:
    if st.session_state.input_text.strip():
        with st.spinner("Model analiz ediyor..."):
            analyzer = load_model()
            start = time.time()
            raw_result = predict_text(analyzer, st.session_state.input_text.strip())
            elapsed = time.time() - start
            st.session_state.analysis_result = parse_result(raw_result)
            st.session_state.analysis_time = elapsed
    else:
        st.session_state.analysis_result = None
        st.session_state.analysis_time = None
        st.warning("Lütfen analiz edilecek bir metin girin.")

st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="panel-title">📘 Örnek Cümleler</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-help">Bir örneğe tıklayarak metin kutusuna aktarabilirsiniz.</div>', unsafe_allow_html=True)

        for i, (emoji, label, sentence) in enumerate(examples):
            button_text = f"{emoji} {label}: {sentence}"

            st.button(
                button_text,
                key=f"example_{i}",
                use_container_width=True,
                type="secondary",
                on_click=use_example,
                args=(sentence,)
            )

with right:
    with st.container(border=True):
        st.markdown('<div class="panel-title">🎯 Tahmin Sonucu</div>', unsafe_allow_html=True)
        render_result(st.session_state.analysis_result, st.session_state.analysis_time)

st.markdown('<div class="footer-text">DuyguPusula | Ahmet Canpolat | 222523024 | BERTurk tabanlı Türkçe duygu analizi projesi</div>', unsafe_allow_html=True)


