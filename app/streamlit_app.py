from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.project_pipeline import (
    ARTIFACTS_DIR,
    CLASSES,
    FIGURES_DIR,
    MAX_SEQUENCE_LEN,
    TABLES_DIR,
    clean_tweet,
    slugify,
)

ARTIFACTS = ROOT / ARTIFACTS_DIR
METADATA_PATH = ARTIFACTS / "metadata.json"
COMPARISON_PATH = ROOT / TABLES_DIR / "comparaison_modeles.csv"
DISTILBERT_MAX_LENGTH = 64


# --------------------------------------------------------------------------- #
# Catalogue des modèles déployables.
# `report_name` = nom exact de la ligne dans comparaison_modeles.csv (sert aussi
# à retrouver la matrice de confusion via slugify).
# --------------------------------------------------------------------------- #
MODEL_CATALOG: dict[str, dict] = {
    "TF-IDF + Régression logistique (déployé)": {
        "kind": "sklearn_pipeline",
        "files": ["selected_tfidf_model.joblib"],
        "report_name": "TF-IDF + Logistic Regression",
    },
    "TF-IDF + ANN (MLP)": {
        "kind": "tfidf_ann",
        "files": ["ann_tfidf.keras", "ann_tfidf_vectorizer.joblib"],
        "report_name": "TF-IDF + ANN",
    },
    "Word2Vec + CNN": {
        "kind": "keras_sequence",
        "files": ["cnn_word2vec.keras", "keras_tokenizer.joblib"],
        "report_name": "Word2Vec + CNN",
    },
    "Word2Vec + BiLSTM": {
        "kind": "keras_sequence",
        "files": ["bilstm_word2vec.keras", "keras_tokenizer.joblib"],
        "report_name": "Word2Vec + BiLSTM",
    },
    "DistilBERT (fine-tuné)": {
        "kind": "distilbert",
        "files": ["distilbert_finetuned/config.json", "distilbert_finetuned/model.safetensors"],
        "report_name": "DistilBERT (fine-tuned)",
    },
}


def available_models() -> dict[str, dict]:
    """Garde uniquement les modèles dont tous les artefacts sont présents."""
    return {
        name: spec
        for name, spec in MODEL_CATALOG.items()
        if all((ARTIFACTS / f).exists() for f in spec["files"])
    }


# --------------------------------------------------------------------------- #
# Chargement + inférence — un prédicteur uniforme par modèle.
# Chaque prédicteur prend une liste de textes nettoyés et renvoie un tableau
# (n, 3) de probabilités, dans l'ordre CLASSES = [negative, neutral, positive].
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Chargement du modèle…")
def get_predictor(name: str):
    spec = MODEL_CATALOG[name]
    kind = spec["kind"]

    if kind == "sklearn_pipeline":
        model = joblib.load(ARTIFACTS / spec["files"][0])

        def predict(texts: list[str]) -> np.ndarray:
            return np.asarray(model.predict_proba(texts))

        return predict

    if kind == "tfidf_ann":
        from src.project_pipeline import import_tensorflow

        _, keras, *_ = import_tensorflow()
        model = keras.models.load_model(ARTIFACTS / spec["files"][0])
        vectorizer = joblib.load(ARTIFACTS / spec["files"][1])

        def predict(texts: list[str]) -> np.ndarray:
            x = vectorizer.transform(texts).astype("float32").toarray()
            return np.asarray(model.predict(x, verbose=0))

        return predict

    if kind == "keras_sequence":
        from src.project_pipeline import import_tensorflow

        _, keras, _, _, pad_sequences = import_tensorflow()
        model = keras.models.load_model(ARTIFACTS / spec["files"][0])
        tokenizer = joblib.load(ARTIFACTS / spec["files"][1])

        def predict(texts: list[str]) -> np.ndarray:
            seqs = tokenizer.texts_to_sequences(texts)
            x = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")
            return np.asarray(model.predict(x, verbose=0))

        return predict

    if kind == "distilbert":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = ARTIFACTS / "distilbert_finetuned"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()

        def predict(texts: list[str]) -> np.ndarray:
            enc = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=DISTILBERT_MAX_LENGTH,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**enc).logits
            return torch.softmax(logits, dim=-1).cpu().numpy()

        return predict

    raise ValueError(f"Type de modèle inconnu : {kind}")


@st.cache_data
def load_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {"classes": CLASSES}


@st.cache_data
def load_comparison() -> pd.DataFrame | None:
    if COMPARISON_PATH.exists():
        return pd.read_csv(COMPARISON_PATH)
    return None


def priority_label(sentiment: str, probability: float, retweets: int) -> str:
    if sentiment == "negative" and (probability >= 0.70 or retweets >= 10):
        return "Priorite haute"
    if sentiment == "negative":
        return "Priorite standard"
    return "Suivi normal"


# --------------------------------------------------------------------------- #
# Onglet 1 — Prédiction
# --------------------------------------------------------------------------- #
def render_prediction(model_name: str, classes: list[str]) -> None:
    tweet = st.text_area(
        "Tweet",
        value="@united my flight is delayed again and nobody can tell me what is happening",
        height=120,
    )
    retweets = st.slider("Nombre de retweets", min_value=0, max_value=100, value=0)

    if st.button("Predire", type="primary"):
        predict = get_predictor(model_name)
        cleaned = clean_tweet(tweet)
        probabilities = predict([cleaned])[0]
        prediction_idx = int(np.argmax(probabilities))
        sentiment = classes[prediction_idx]
        confidence = float(probabilities[prediction_idx])

        # Mémorise pour les stats de session.
        st.session_state.setdefault("history", []).append(sentiment)

        cols = st.columns(3)
        cols[0].metric("Sentiment", sentiment)
        cols[1].metric("Confiance", f"{confidence:.1%}")
        cols[2].metric("Priorisation", priority_label(sentiment, confidence, retweets))

        chart = pd.DataFrame(
            {"sentiment": classes, "probabilite": np.asarray(probabilities, dtype=float)}
        ).set_index("sentiment")
        st.bar_chart(chart)

        with st.expander("Texte nettoye"):
            st.write(cleaned or "_(vide après nettoyage)_")

    history = st.session_state.get("history", [])
    if history:
        st.divider()
        st.caption(f"Statistiques de la session — {len(history)} prediction(s)")
        counts = Counter(history)
        cols = st.columns(len(classes))
        for col, label in zip(cols, classes):
            col.metric(label, counts.get(label, 0))
        if st.button("Reinitialiser les statistiques de session"):
            st.session_state["history"] = []
            st.rerun()


# --------------------------------------------------------------------------- #
# Onglet 2 — Statistiques
# --------------------------------------------------------------------------- #
def render_stats(model_name: str) -> None:
    comparison = load_comparison()
    spec = MODEL_CATALOG[model_name]
    report_name = spec["report_name"]

    st.subheader("Comparaison des modeles (jeu de test fige)")
    if comparison is None:
        st.info("Tableau de comparaison introuvable. Lancez d'abord : python -m src.project_pipeline")
    else:
        view = comparison[["model", "accuracy", "macro_f1", "weighted_f1"]].copy()
        view = view.sort_values("macro_f1", ascending=False).reset_index(drop=True)
        st.dataframe(
            view.style.format({"accuracy": "{:.3f}", "macro_f1": "{:.3f}", "weighted_f1": "{:.3f}"}),
            use_container_width=True,
        )
        st.bar_chart(view.set_index("model")[["macro_f1", "accuracy"]])

        st.subheader(f"Detail du modele selectionne : {model_name}")
        row = comparison[comparison["model"] == report_name]
        if row.empty:
            st.info("Pas de metriques enregistrees pour ce modele dans le tableau de comparaison.")
        else:
            row = row.iloc[0]
            cols = st.columns(3)
            cols[0].metric("Accuracy", f"{row['accuracy']:.3f}")
            cols[1].metric("F1-macro", f"{row['macro_f1']:.3f}")
            cols[2].metric("F1-pondere", f"{row['weighted_f1']:.3f}")

            per_class = pd.DataFrame(
                {
                    "classe": CLASSES,
                    "precision": [row[f"{c}_precision"] for c in CLASSES],
                    "rappel": [row[f"{c}_recall"] for c in CLASSES],
                    "f1": [row[f"{c}_f1"] for c in CLASSES],
                    "support": [int(row[f"{c}_support"]) for c in CLASSES],
                }
            ).set_index("classe")
            st.dataframe(
                per_class.style.format(
                    {"precision": "{:.3f}", "rappel": "{:.3f}", "f1": "{:.3f}"}
                ),
                use_container_width=True,
            )

    confusion_path = ROOT / FIGURES_DIR / f"confusion_{slugify(report_name)}.png"
    distribution_path = ROOT / FIGURES_DIR / "01_distribution_sentiments.png"
    cols = st.columns(2)
    if confusion_path.exists():
        cols[0].image(str(confusion_path), caption="Matrice de confusion", use_container_width=True)
    if distribution_path.exists():
        cols[1].image(str(distribution_path), caption="Distribution du dataset", use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Airline Tweet Sentiment", page_icon="T", layout="centered")
    st.title("Analyse de sentiment des tweets aeriens")

    models = available_models()
    if not models:
        st.error("Aucun modele trouve dans artifacts/. Lancez d'abord : python -m src.project_pipeline")
        st.stop()

    metadata = load_metadata()
    classes = metadata.get("classes", CLASSES)

    with st.sidebar:
        st.header("Modele")
        model_name = st.selectbox("Choisir le modele", list(models.keys()))
        st.caption(
            "Seuls les modeles dont les artefacts existent dans `artifacts/` sont proposes. "
            "Les modeles Keras / DistilBERT peuvent etre plus longs a charger la premiere fois."
        )

    tab_predict, tab_stats = st.tabs(["Prediction", "Statistiques"])
    with tab_predict:
        render_prediction(model_name, classes)
    with tab_stats:
        render_stats(model_name)


if __name__ == "__main__":
    main()
