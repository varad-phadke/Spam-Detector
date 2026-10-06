"""
app.py — Spam Detection (self-contained Streamlit app)
-------------------------------------------------------
Trains a TF-IDF + Logistic Regression model at startup (cached),
then classifies messages entered by the user.

No external backend required — runs entirely inside Streamlit Cloud.
"""

import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        font-size: 1.15rem;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .spam-box { background: #ffe4e4; color: #b91c1c; border: 2px solid #f87171; }
    .ham-box  { background: #d1fae5; color: #065f46; border: 2px solid #34d399; }
    .conf-text { font-size: 0.95rem; font-weight: 400; margin-top: 0.4rem; }
</style>
""", unsafe_allow_html=True)


# ── Text cleaner ───────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u0900-\u097f\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Load data & train model (cached — runs once per session) ───────────────────
@st.cache_resource(show_spinner="Training model…")
def load_model():
    df = pd.read_csv("spam_detection_dataset.csv", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df[["Message", "Label"]].dropna()
    df["Message"] = df["Message"].astype(str).apply(clean_text)
    df["Label"]   = df["Label"].str.strip().str.capitalize()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])
    pipeline.fit(df["Message"], df["Label"])
    return pipeline


# ── Prediction helper ──────────────────────────────────────────────────────────
def predict(model: Pipeline, message: str) -> tuple[str, float]:
    """Returns (label, confidence) for the given raw message."""
    cleaned    = clean_text(message)
    label      = model.predict([cleaned])[0]
    proba      = model.predict_proba([cleaned])[0]
    classes    = list(model.classes_)
    confidence = float(proba[classes.index(label)])
    return label, round(confidence, 4)


# ── Load model ─────────────────────────────────────────────────────────────────
model = load_model()


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🛡️ Spam Message Detector")
st.markdown(
    "Enter any text message below and the model will classify it as "
    "**Ham** ✅ (legitimate) or **Spam** 🚫 (unwanted / fraudulent / promotional)."
)
st.divider()


# ── Input form ─────────────────────────────────────────────────────────────────
with st.form("predict_form", clear_on_submit=False):
    user_message = st.text_area(
        "📩 Your Message",
        placeholder="e.g. Congratulations! You've won a prize. Claim now!",
        height=130,
    )
    submitted = st.form_submit_button("🔍 Classify Message", use_container_width=True)


# ── Result ─────────────────────────────────────────────────────────────────────
if submitted:
    if not user_message.strip():
        st.warning("Please enter a message before classifying.")
    else:
        label, confidence = predict(model, user_message)

        if label == "Spam":
            st.markdown(
                f'<div class="result-box spam-box">'
                f'🚫 SPAM<br>'
                f'<span class="conf-text">Confidence: {confidence * 100:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-box ham-box">'
                f'✅ HAM &nbsp;(Legitimate)<br>'
                f'<span class="conf-text">Confidence: {confidence * 100:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Confidence columns
        st.markdown("#### Prediction Confidence")
        col1, col2 = st.columns(2)
        ham_pct  = (1 - confidence) * 100 if label == "Spam" else confidence * 100
        spam_pct = confidence * 100        if label == "Spam" else (1 - confidence) * 100
        with col1:
            st.metric("Ham  ✅", f"{ham_pct:.1f}%")
        with col2:
            st.metric("Spam 🚫", f"{spam_pct:.1f}%")

        with st.expander("📋 Details"):
            st.json({
                "message":    user_message,
                "label":      label,
                "confidence": confidence,
            })


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("💡 Try These Examples")
    examples = {
        "🚫 Spam — Prize claim":    "Congratulations! You have won a Rs.50,000 cash prize. Claim now!",
        "🚫 Spam — Bank urgent":    "URGENT! Your bank account will be blocked. Verify your details immediately.",
        "🚫 Spam — Work from home": "Earn Rs.5,000 daily from home with zero investment. Join now!",
        "✅ Ham  — College friend":  "Hey, are we meeting at college tomorrow?",
        "✅ Ham  — OTP message":     "Your OTP for login is 483921. Do not share it with anyone.",
        "✅ Ham  — Ticket booked":   "Your train ticket has been successfully booked for Monday.",
    }
    for btn_label, text in examples.items():
        if st.button(btn_label, use_container_width=True):
            st.info(f"Copy this into the text box above:\n\n*{text}*")

    st.divider()
    st.markdown("#### ℹ️ About")
    st.markdown(
        "**Model:** TF-IDF (1-2 grams) + Logistic Regression  \n"
        "**Trained on:** 20 labelled messages  \n"
        "**Deployment:** Streamlit Community Cloud  \n"
    )

st.divider()
st.caption("Spam Detector · Built with Python & Streamlit")
