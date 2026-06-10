from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import streamlit as st

from src.project_pipeline import CLASSES, clean_tweet

MODEL_PATH = ROOT / "artifacts" / "selected_tfidf_model.joblib"
METADATA_PATH = ROOT / "artifacts" / "metadata.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {"classes": CLASSES}


def priority_label(sentiment: str, probability: float, retweets: int) -> str:
    if sentiment == "negative" and (probability >= 0.70 or retweets >= 10):
        return "Priorite haute"
    if sentiment == "negative":
        return "Priorite standard"
    return "Suivi normal"


def main() -> None:
    st.set_page_config(page_title="Airline Tweet Sentiment", page_icon="T", layout="centered")
    st.title("Analyse de sentiment des tweets aeriens")

    if not MODEL_PATH.exists():
        st.error("Modele introuvable. Lancez d'abord: python -m src.project_pipeline")
        st.stop()

    model = load_model()
    metadata = load_metadata()
    classes = metadata.get("classes", CLASSES)

    tweet = st.text_area(
        "Tweet",
        value="@united my flight is delayed again and nobody can tell me what is happening",
        height=120,
    )
    retweets = st.slider("Nombre de retweets", min_value=0, max_value=100, value=0)

    if st.button("Predire", type="primary"):
        cleaned = clean_tweet(tweet)
        probabilities = model.predict_proba([cleaned])[0]
        prediction_idx = int(probabilities.argmax())
        sentiment = classes[prediction_idx]
        confidence = float(probabilities[prediction_idx])

        cols = st.columns(3)
        cols[0].metric("Sentiment", sentiment)
        cols[1].metric("Confiance", f"{confidence:.1%}")
        cols[2].metric("Priorisation", priority_label(sentiment, confidence, retweets))

        chart = pd.DataFrame({"sentiment": classes, "probabilite": probabilities}).set_index("sentiment")
        st.bar_chart(chart)

        with st.expander("Texte nettoye"):
            st.write(cleaned)


if __name__ == "__main__":
    main()
