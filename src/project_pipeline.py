from __future__ import annotations

import html
import json
import os
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
DATA_PATH = Path("tweets/Tweets.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")

TEXT_COL = "text"
TARGET_COL = "airline_sentiment"
CLASSES = ["negative", "neutral", "positive"]
CLASS_TO_ID = {label: idx for idx, label in enumerate(CLASSES)}
ID_TO_CLASS = {idx: label for label, idx in CLASS_TO_ID.items()}

MAX_TFIDF_FEATURES = 20_000
ANN_TFIDF_FEATURES = 5_000
MAX_VOCAB = 10_000
MAX_SEQUENCE_LEN = 45
EMBEDDING_DIM = 100


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            tf.config.threading.set_intra_op_parallelism_threads(4)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:
            pass
    except Exception:
        pass


def ensure_dirs() -> None:
    for path in [PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, ARTIFACTS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_tweet(text: object) -> str:
    """Normalize Twitter-specific noise while keeping useful lexical signal."""
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"@\w+", " usertoken ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"&amp;", " and ", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"\d+", " numbertoken ", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"[^a-z_\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep_cols = [
        "tweet_id",
        "airline_sentiment",
        "airline_sentiment_confidence",
        "negativereason",
        "negativereason_confidence",
        "airline",
        "retweet_count",
        "text",
        "tweet_created",
        "tweet_location",
        "user_timezone",
    ]
    df = df[[col for col in keep_cols if col in df.columns]].copy()
    df = df.dropna(subset=[TEXT_COL, TARGET_COL])
    df = df[df[TARGET_COL].isin(CLASSES)].copy()
    df["clean_text"] = df[TEXT_COL].map(clean_tweet)
    # Drop empty cleaned texts and de-duplicate on cleaned text BEFORE any split.
    # Identical tweets (retweets, canned replies) otherwise leak across train/test
    # and inflate the reported scores. Keeping the first occurrence also resolves
    # the few duplicates that carry conflicting sentiment labels.
    df = df[df["clean_text"].str.len() > 0].copy()
    df = df.drop_duplicates(subset="clean_text", keep="first").reset_index(drop=True)
    df["tweet_length"] = df[TEXT_COL].astype(str).str.len()
    df["clean_token_count"] = df["clean_text"].str.split().map(len)
    df["label_id"] = df[TARGET_COL].map(CLASS_TO_ID)
    df["tweet_created"] = pd.to_datetime(df["tweet_created"], errors="coerce", utc=True)
    return df


def create_splits(df: pd.DataFrame) -> DatasetSplits:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET_COL],
    )
    splits = DatasetSplits(
        train=train_df.reset_index(drop=True),
        validation=validation_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )
    splits.train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    splits.validation.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    splits.test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    return splits


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET_COL].value_counts().reindex(CLASSES)
    percents = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = ["#d95f59", "#6a8fbf", "#4c9f70"]
    bars = ax.bar(counts.index, counts.values, color=palette)
    for bar, percent in zip(bars, percents):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 70,
            f"{percent:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_title("Distribution des sentiments")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Nombre de tweets")
    ax.set_ylim(0, counts.max() * 1.16)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_distribution_sentiments.png", dpi=160)
    plt.close(fig)


def plot_airline_sentiment(df: pd.DataFrame) -> None:
    table = pd.crosstab(df["airline"], df[TARGET_COL], normalize="index").reindex(columns=CLASSES)
    fig, ax = plt.subplots(figsize=(10, 6))
    table.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=["#d95f59", "#6a8fbf", "#4c9f70"],
        width=0.75,
    )
    ax.set_title("Part de sentiments par compagnie")
    ax.set_xlabel("Part des tweets")
    ax.set_ylabel("Compagnie")
    ax.legend(title="Sentiment", loc="lower right")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_sentiments_par_compagnie.png", dpi=160)
    plt.close(fig)


def plot_negative_reasons(df: pd.DataFrame) -> None:
    reasons = df.loc[df[TARGET_COL] == "negative", "negativereason"].fillna("Unknown")
    counts = reasons.value_counts().head(10).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(counts.index, counts.values, color="#d95f59")
    ax.set_title("Principaux motifs de tweets negatifs")
    ax.set_xlabel("Nombre de tweets")
    ax.set_ylabel("Motif")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_motifs_negatifs.png", dpi=160)
    plt.close(fig)


def plot_tweet_lengths(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df, x=TARGET_COL, y="tweet_length", order=CLASSES, ax=axes[0], palette="Set2")
    axes[0].set_title("Longueur brute par sentiment")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Nombre de caracteres")

    sns.histplot(
        data=df,
        x="clean_token_count",
        hue=TARGET_COL,
        hue_order=CLASSES,
        bins=30,
        kde=False,
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[1],
        palette=["#d95f59", "#6a8fbf", "#4c9f70"],
    )
    axes[1].set_title("Nombre de tokens apres nettoyage")
    axes[1].set_xlabel("Tokens")
    axes[1].set_ylabel("Densite")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_longueurs_tweets.png", dpi=160)
    plt.close(fig)


def _top_terms(texts: Iterable[str], n_terms: int = 30) -> pd.DataFrame:
    vectorizer = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS),
        min_df=2,
        token_pattern=r"(?u)\b[a-z_][a-z_]+\b",
    )
    matrix = vectorizer.fit_transform(texts)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    order = counts.argsort()[::-1][:n_terms]
    return pd.DataFrame({"term": terms[order], "count": counts[order]})


def plot_wordcloud_like(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, sentiment, color in zip(axes, CLASSES, ["#d95f59", "#6a8fbf", "#4c9f70"]):
        terms = _top_terms(df.loc[df[TARGET_COL] == sentiment, "clean_text"], n_terms=32)
        ax.set_title(f"Nuage de mots - {sentiment}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        if terms.empty:
            continue
        max_count = terms["count"].max()
        positions = [
            (0.50, 0.52),
            (0.24, 0.70),
            (0.72, 0.70),
            (0.24, 0.34),
            (0.72, 0.34),
            (0.50, 0.82),
            (0.50, 0.22),
            (0.12, 0.52),
            (0.88, 0.52),
            (0.34, 0.58),
            (0.66, 0.58),
            (0.34, 0.46),
            (0.66, 0.46),
            (0.15, 0.82),
            (0.85, 0.82),
            (0.15, 0.18),
            (0.85, 0.18),
            (0.38, 0.75),
            (0.62, 0.75),
            (0.38, 0.29),
            (0.62, 0.29),
            (0.08, 0.66),
            (0.92, 0.66),
            (0.08, 0.38),
            (0.92, 0.38),
            (0.28, 0.88),
            (0.72, 0.88),
            (0.28, 0.12),
            (0.72, 0.12),
            (0.50, 0.66),
            (0.50, 0.38),
            (0.50, 0.10),
        ]
        for (_, row), (x, y) in zip(terms.iterrows(), positions):
            size = 9 + 23 * (row["count"] / max_count)
            ax.text(x, y, row["term"], ha="center", va="center", fontsize=size, color=color, alpha=0.92)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_nuages_mots.png", dpi=160)
    plt.close(fig)


def generate_eda(df: pd.DataFrame) -> None:
    plot_class_distribution(df)
    plot_airline_sentiment(df)
    plot_negative_reasons(df)
    plot_tweet_lengths(df)
    plot_wordcloud_like(df)

    airline_summary = (
        pd.crosstab(df["airline"], df[TARGET_COL], normalize="index")
        .reindex(columns=CLASSES)
        .sort_values("negative", ascending=False)
    )
    airline_summary.to_csv(TABLES_DIR / "sentiments_par_compagnie.csv")

    negative_reasons = df.loc[df[TARGET_COL] == "negative", "negativereason"].value_counts()
    negative_reasons.to_csv(TABLES_DIR / "motifs_negatifs.csv", header=["count"])


def label_arrays(splits: DatasetSplits) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_train = splits.train["label_id"].to_numpy()
    y_validation = splits.validation["label_id"].to_numpy()
    y_test = splits.test["label_id"].to_numpy()
    return y_train, y_validation, y_test


def class_weights(y_train: np.ndarray) -> dict[int, float]:
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(CLASSES)), y=y_train)
    return {int(label): float(weight) for label, weight in zip(np.arange(len(CLASSES)), weights)}


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(CLASSES)),
        zero_division=0,
    )
    result = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    for i, label in enumerate(CLASSES):
        result[f"{label}_precision"] = float(precision[i])
        result[f"{label}_recall"] = float(recall[i])
        result[f"{label}_f1"] = float(f1[i])
        result[f"{label}_support"] = int(support[i])

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(CLASSES)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    with (TABLES_DIR / f"classification_report_{slugify(model_name)}.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    return result


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def save_confusion_matrix(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar_kws={"label": "Taux normalise par vraie classe"},
        ax=ax,
    )
    ax.set_title(f"Matrice de confusion - {model_name}")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Vraie classe")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_{slugify(model_name)}.png", dpi=160)
    plt.close(fig)


def save_learning_curve(history: object, model_name: str) -> None:
    if history is None:
        return
    values = pd.DataFrame(history.history)
    if values.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if "loss" in values:
        axes[0].plot(values.index + 1, values["loss"], marker="o", label="train")
    if "val_loss" in values:
        axes[0].plot(values.index + 1, values["val_loss"], marker="o", label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if "accuracy" in values:
        axes[1].plot(values.index + 1, values["accuracy"], marker="o", label="train")
    if "val_accuracy" in values:
        axes[1].plot(values.index + 1, values["val_accuracy"], marker="o", label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle(f"Courbes d'apprentissage - {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"learning_curve_{slugify(model_name)}.png", dpi=160)
    plt.close(fig)
    values.to_csv(TABLES_DIR / f"history_{slugify(model_name)}.csv", index=False)


def train_logistic_tfidf(splits: DatasetSplits) -> tuple[Pipeline, dict[str, object], np.ndarray]:
    y_train, _, y_test = label_arrays(splits)
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=MAX_TFIDF_FEATURES,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b[a-z_][a-z_]+\b",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    C=2.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(splits.train["clean_text"], y_train)
    y_pred = model.predict(splits.test["clean_text"])
    y_proba = model.predict_proba(splits.test["clean_text"])
    result = evaluate_predictions("TF-IDF + Logistic Regression", y_test, y_pred, y_proba)
    save_confusion_matrix("TF-IDF + Logistic Regression", y_test, y_pred)
    joblib.dump(model, ARTIFACTS_DIR / "tfidf_logistic_regression.joblib")
    save_top_tfidf_terms(model)
    return model, result, y_pred


def save_top_tfidf_terms(model: Pipeline, n_terms: int = 20) -> None:
    vectorizer: TfidfVectorizer = model.named_steps["tfidf"]
    classifier: LogisticRegression = model.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (label, ax) in enumerate(zip(CLASSES, axes)):
        coefs = classifier.coef_[i]
        order = np.argsort(coefs)[-n_terms:]
        terms = feature_names[order]
        weights = coefs[order]
        rows.extend({"class": label, "term": term, "weight": float(weight)} for term, weight in zip(terms, weights))
        ax.barh(terms, weights, color=["#d95f59", "#6a8fbf", "#4c9f70"][i])
        ax.set_title(label)
        ax.set_xlabel("Poids logistique")
    fig.suptitle("Termes les plus influents - TF-IDF + regression logistique")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "interpretabilite_top_tfidf_terms.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(TABLES_DIR / "top_tfidf_terms.csv", index=False)


def import_tensorflow():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    return tf, keras, layers, Tokenizer, pad_sequences


def train_ann_tfidf(splits: DatasetSplits, weights: dict[int, float]) -> tuple[object, dict[str, object], np.ndarray]:
    _, keras, layers, _, _ = import_tensorflow()
    y_train, y_validation, y_test = label_arrays(splits)

    vectorizer = TfidfVectorizer(
        max_features=ANN_TFIDF_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-z_][a-z_]+\b",
    )
    x_train = vectorizer.fit_transform(splits.train["clean_text"]).astype("float32").toarray()
    x_validation = vectorizer.transform(splits.validation["clean_text"]).astype("float32").toarray()
    x_test = vectorizer.transform(splits.test["clean_text"]).astype("float32").toarray()

    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1],)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.35),
            layers.Dense(96, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(len(CLASSES), activation="softmax"),
        ],
        name="ann_tfidf",
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=8,
        batch_size=128,
        class_weight=weights,
        callbacks=callbacks,
        verbose=2,
    )
    y_proba = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = y_proba.argmax(axis=1)
    result = evaluate_predictions("TF-IDF + ANN", y_test, y_pred, y_proba)
    save_confusion_matrix("TF-IDF + ANN", y_test, y_pred)
    save_learning_curve(history, "TF-IDF + ANN")
    joblib.dump(vectorizer, ARTIFACTS_DIR / "ann_tfidf_vectorizer.joblib")
    model.save(ARTIFACTS_DIR / "ann_tfidf.keras")
    return model, result, y_pred


def build_tokenizer_and_sequences(splits: DatasetSplits):
    _, _, _, Tokenizer, pad_sequences = import_tensorflow()
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="oovtoken", filters="")
    tokenizer.fit_on_texts(splits.train["clean_text"])
    sequences_train = tokenizer.texts_to_sequences(splits.train["clean_text"])
    sequences_validation = tokenizer.texts_to_sequences(splits.validation["clean_text"])
    sequences_test = tokenizer.texts_to_sequences(splits.test["clean_text"])
    x_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")
    x_validation = pad_sequences(sequences_validation, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")
    x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")
    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
    joblib.dump(tokenizer, ARTIFACTS_DIR / "keras_tokenizer.joblib")
    metadata = {
        "max_vocab": MAX_VOCAB,
        "vocab_size": vocab_size,
        "max_sequence_len": MAX_SEQUENCE_LEN,
        "embedding_dim": EMBEDDING_DIM,
    }
    with (ARTIFACTS_DIR / "tokenizer_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return tokenizer, vocab_size, x_train, x_validation, x_test


def build_skipgram_pairs(
    sequences: list[list[int]],
    vocab_size: int,
    window_size: int = 2,
    negative_samples: int = 2,
    max_positive_pairs: int = 180_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    # Shuffle the corpus before collecting pairs: the max_positive_pairs cap stops
    # iteration early, so without this the embeddings would only ever see the first
    # tweets (in their original order). Shuffling first makes the cap a uniform
    # sample across the whole training corpus.
    sequences = list(sequences)
    rng.shuffle(sequences)
    positives: list[tuple[int, int]] = []
    for sequence in sequences:
        tokens = [int(token) for token in sequence if 1 < token < vocab_size]
        if len(tokens) < 2:
            continue
        for i, target in enumerate(tokens):
            left = max(0, i - window_size)
            right = min(len(tokens), i + window_size + 1)
            for j in range(left, right):
                if i == j:
                    continue
                positives.append((target, tokens[j]))
                if len(positives) >= max_positive_pairs:
                    break
            if len(positives) >= max_positive_pairs:
                break
        if len(positives) >= max_positive_pairs:
            break
    if not positives:
        raise ValueError("No skip-gram pairs were generated.")

    rng.shuffle(positives)
    n_pos = len(positives)
    total = n_pos * (1 + negative_samples)
    targets = np.empty(total, dtype="int32")
    contexts = np.empty(total, dtype="int32")
    labels = np.empty(total, dtype="float32")

    cursor = 0
    for target, context in positives:
        targets[cursor] = target
        contexts[cursor] = context
        labels[cursor] = 1.0
        cursor += 1
        for _ in range(negative_samples):
            targets[cursor] = target
            contexts[cursor] = rng.integers(2, vocab_size, dtype="int32")
            labels[cursor] = 0.0
            cursor += 1

    order = rng.permutation(total)
    return targets[order], contexts[order], labels[order]


def train_word2vec_embeddings(
    splits: DatasetSplits,
    tokenizer: object,
    vocab_size: int,
) -> np.ndarray:
    _, keras, layers, _, _ = import_tensorflow()
    train_sequences = tokenizer.texts_to_sequences(splits.train["clean_text"])
    targets, contexts, labels = build_skipgram_pairs(train_sequences, vocab_size)

    target_input = layers.Input(shape=(), dtype="int32", name="target")
    context_input = layers.Input(shape=(), dtype="int32", name="context")
    target_embedding = layers.Embedding(vocab_size, EMBEDDING_DIM, name="target_embedding")(target_input)
    context_embedding = layers.Embedding(vocab_size, EMBEDDING_DIM, name="context_embedding")(context_input)
    score = layers.Dot(axes=-1)([target_embedding, context_embedding])
    score = layers.Activation("sigmoid")(score)
    model = keras.Model([target_input, context_input], score, name="word2vec_skipgram")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-3), loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit([targets, contexts], labels, epochs=3, batch_size=1024, validation_split=0.10, verbose=2)
    save_learning_curve(history, "Word2Vec skip-gram")
    embeddings = model.get_layer("target_embedding").get_weights()[0]
    embeddings[0] = 0.0
    np.save(ARTIFACTS_DIR / "word2vec_embeddings.npy", embeddings)
    save_word2vec_neighbors(tokenizer, embeddings, vocab_size)
    return embeddings


def save_word2vec_neighbors(tokenizer: object, embeddings: np.ndarray, vocab_size: int) -> None:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-8)
    seeds = ["flight", "delay", "cancelled", "service", "bag", "customer", "thanks", "late"]
    rows = []
    index_word = {idx: word for word, idx in tokenizer.word_index.items() if idx < vocab_size}
    for seed in seeds:
        idx = tokenizer.word_index.get(seed)
        if idx is None or idx >= vocab_size:
            continue
        similarities = normalized @ normalized[idx]
        similarities[idx] = -1
        for neighbor_idx in similarities.argsort()[::-1][:8]:
            rows.append(
                {
                    "seed": seed,
                    "neighbor": index_word.get(int(neighbor_idx), ""),
                    "similarity": float(similarities[neighbor_idx]),
                }
            )
    pd.DataFrame(rows).to_csv(TABLES_DIR / "word2vec_voisins.csv", index=False)


def build_embedding_layer(layers, embeddings: np.ndarray, trainable: bool = True, mask_zero: bool = True):
    return layers.Embedding(
        input_dim=embeddings.shape[0],
        output_dim=embeddings.shape[1],
        weights=[embeddings],
        mask_zero=mask_zero,
        trainable=trainable,
        name="word2vec_embedding",
    )


def train_cnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    embeddings: np.ndarray,
    weights: dict[int, float],
) -> tuple[object, dict[str, object], np.ndarray]:
    _, keras, layers, _, _ = import_tensorflow()
    inputs = layers.Input(shape=(MAX_SEQUENCE_LEN,), dtype="int32")
    x = build_embedding_layer(layers, embeddings, mask_zero=False)(inputs)
    x = layers.Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(96, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="cnn_word2vec")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=8,
        batch_size=128,
        class_weight=weights,
        callbacks=callbacks,
        verbose=2,
    )
    y_proba = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = y_proba.argmax(axis=1)
    result = evaluate_predictions("Word2Vec + CNN", y_test, y_pred, y_proba)
    save_confusion_matrix("Word2Vec + CNN", y_test, y_pred)
    save_learning_curve(history, "Word2Vec + CNN")
    model.save(ARTIFACTS_DIR / "cnn_word2vec.keras")
    return model, result, y_pred


def train_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    embeddings: np.ndarray,
    weights: dict[int, float],
) -> tuple[object, dict[str, object], np.ndarray]:
    _, keras, layers, _, _ = import_tensorflow()
    inputs = layers.Input(shape=(MAX_SEQUENCE_LEN,), dtype="int32")
    x = build_embedding_layer(layers, embeddings)(inputs)
    x = layers.Bidirectional(layers.LSTM(64, dropout=0.20, recurrent_dropout=0.10))(x)
    x = layers.Dense(80, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="bilstm_word2vec")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=8e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=8,
        batch_size=128,
        class_weight=weights,
        callbacks=callbacks,
        verbose=2,
    )
    y_proba = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = y_proba.argmax(axis=1)
    result = evaluate_predictions("Word2Vec + BiLSTM", y_test, y_pred, y_proba)
    save_confusion_matrix("Word2Vec + BiLSTM", y_test, y_pred)
    save_learning_curve(history, "Word2Vec + BiLSTM")
    model.save(ARTIFACTS_DIR / "bilstm_word2vec.keras")
    return model, result, y_pred


def train_attention(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    embeddings: np.ndarray,
    weights: dict[int, float],
    tokenizer: object,
    test_df: pd.DataFrame,
) -> tuple[object, dict[str, object], np.ndarray]:
    tf, keras, layers, _, _ = import_tensorflow()

    class AttentionPooling(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.supports_masking = True
            self.score_dense = layers.Dense(1)

        def call(self, inputs, mask=None):
            scores = tf.squeeze(self.score_dense(tf.tanh(inputs)), axis=-1)
            if mask is not None:
                scores = tf.where(mask, scores, tf.fill(tf.shape(scores), tf.constant(-1e9, dtype=scores.dtype)))
            weights_ = tf.nn.softmax(scores, axis=1)
            context = tf.reduce_sum(inputs * tf.expand_dims(weights_, axis=-1), axis=1)
            return context, weights_

        def compute_mask(self, inputs, mask=None):
            return None

    inputs = layers.Input(shape=(MAX_SEQUENCE_LEN,), dtype="int32")
    x = build_embedding_layer(layers, embeddings)(inputs)
    sequence = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.20, recurrent_dropout=0.10))(x)
    context, attention_weights = AttentionPooling(name="attention_pooling")(sequence)
    x = layers.Dense(80, activation="relu")(context)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="attention_bilstm_word2vec")
    attention_model = keras.Model(inputs, attention_weights, name="attention_weights_model")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=8e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=8,
        batch_size=128,
        class_weight=weights,
        callbacks=callbacks,
        verbose=2,
    )
    y_proba = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = y_proba.argmax(axis=1)
    result = evaluate_predictions("Word2Vec + BiLSTM Attention", y_test, y_pred, y_proba)
    save_confusion_matrix("Word2Vec + BiLSTM Attention", y_test, y_pred)
    save_learning_curve(history, "Word2Vec + BiLSTM Attention")
    save_attention_plot(attention_model, x_test, y_test, y_pred, tokenizer, test_df)
    return model, result, y_pred


def save_attention_plot(
    attention_model: object,
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    tokenizer: object,
    test_df: pd.DataFrame,
) -> None:
    correct_negative = np.where((y_test == CLASS_TO_ID["negative"]) & (y_pred == y_test))[0]
    sample_idx = int(correct_negative[0]) if len(correct_negative) else 0
    weights = attention_model.predict(x_test[[sample_idx]], verbose=0)[0]
    tokens = [tokenizer.index_word.get(int(idx), "") for idx in x_test[sample_idx] if int(idx) != 0]
    weights = weights[: len(tokens)]
    if not tokens:
        return
    order = np.argsort(weights)[::-1][: min(12, len(tokens))]
    selected = pd.DataFrame({"token": np.array(tokens)[order], "attention": weights[order]}).iloc[::-1]
    selected.to_csv(TABLES_DIR / "attention_exemple.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(selected["token"], selected["attention"], color="#6a8fbf")
    ax.set_title("Poids d'attention sur un tweet negatif bien classe")
    ax.set_xlabel("Poids d'attention")
    ax.set_ylabel("Token")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "interpretabilite_attention_exemple.png", dpi=160)
    plt.close(fig)

    with (TABLES_DIR / "attention_exemple_texte.txt").open("w", encoding="utf-8") as file:
        file.write(str(test_df.iloc[sample_idx][TEXT_COL]))


def train_deep_models(splits: DatasetSplits, weights: dict[int, float]) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    tokenizer, vocab_size, x_train, x_validation, x_test = build_tokenizer_and_sequences(splits)
    embeddings = train_word2vec_embeddings(splits, tokenizer, vocab_size)
    y_train, y_validation, y_test = label_arrays(splits)

    results = []
    predictions = {}
    _, cnn_result, cnn_pred = train_cnn(x_train, y_train, x_validation, y_validation, x_test, y_test, embeddings, weights)
    results.append(cnn_result)
    predictions["Word2Vec + CNN"] = cnn_pred

    _, lstm_result, lstm_pred = train_lstm(x_train, y_train, x_validation, y_validation, x_test, y_test, embeddings, weights)
    results.append(lstm_result)
    predictions["Word2Vec + BiLSTM"] = lstm_pred

    _, attention_result, attention_pred = train_attention(
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        embeddings,
        weights,
        tokenizer,
        splits.test,
    )
    results.append(attention_result)
    predictions["Word2Vec + BiLSTM Attention"] = attention_pred
    return results, predictions


def save_error_analysis(test_df: pd.DataFrame, predictions: dict[str, np.ndarray]) -> None:
    y_test = test_df["label_id"].to_numpy()
    rows = []
    for model_name, y_pred in predictions.items():
        mismatches = np.where(y_pred != y_test)[0]
        for idx in mismatches[:60]:
            rows.append(
                {
                    "model": model_name,
                    "true_label": ID_TO_CLASS[int(y_test[idx])],
                    "predicted_label": ID_TO_CLASS[int(y_pred[idx])],
                    "text": test_df.iloc[idx][TEXT_COL],
                    "clean_text": test_df.iloc[idx]["clean_text"],
                    "airline": test_df.iloc[idx].get("airline", ""),
                    "negativereason": test_df.iloc[idx].get("negativereason", ""),
                    "confidence": test_df.iloc[idx].get("airline_sentiment_confidence", np.nan),
                }
            )
    pd.DataFrame(rows).to_csv(TABLES_DIR / "analyse_erreurs_echantillon.csv", index=False)

    neutral_rows = []
    for model_name, y_pred in predictions.items():
        neutral_idx = np.where(y_test == CLASS_TO_ID["neutral"])[0]
        neutral_errors = neutral_idx[y_pred[neutral_idx] != y_test[neutral_idx]]
        for idx in neutral_errors[:40]:
            neutral_rows.append(
                {
                    "model": model_name,
                    "predicted_label": ID_TO_CLASS[int(y_pred[idx])],
                    "text": test_df.iloc[idx][TEXT_COL],
                    "clean_text": test_df.iloc[idx]["clean_text"],
                    "airline": test_df.iloc[idx].get("airline", ""),
                    "confidence": test_df.iloc[idx].get("airline_sentiment_confidence", np.nan),
                }
            )
    pd.DataFrame(neutral_rows).to_csv(TABLES_DIR / "analyse_erreurs_neutre.csv", index=False)


def write_model_card(results_df: pd.DataFrame, df: pd.DataFrame, splits: DatasetSplits, deploy_model: str) -> None:
    class_dist = df[TARGET_COL].value_counts(normalize=True).reindex(CLASSES) * 100
    best = results_df.sort_values("macro_f1", ascending=False).iloc[0]
    content = f"""# Projet NLP - Sentiment des tweets aeriens

## Problematique business

Comment classer automatiquement les tweets adresses aux compagnies aeriennes en `negative`, `neutral` ou `positive` afin de prioriser les reponses du service client, suivre la satisfaction par compagnie et detecter rapidement les pics d'insatisfaction ?

Questions mesurables:

1. Quelle compagnie concentre la plus forte part de tweets negatifs ?
2. Quels motifs expliquent principalement l'insatisfaction ?
3. Quel compromis modele / cout de calcul donne le meilleur F1-macro pour une demonstration deployable ?

## Protocole

- Donnees: {len(df):,} tweets, split stratifie fige en train ({len(splits.train):,}), validation ({len(splits.validation):,}) et test ({len(splits.test):,}).
- Metrique principale: F1-macro, car les classes sont desequilibrees.
- Metriques secondaires: accuracy, F1 par classe, matrices de confusion et analyse d'erreurs.
- Desequilibre: `class_weight=balanced` pour les modeles supervises et lecture prioritaire du F1-macro.
- Nettoyage: minuscules, URLs et mentions remplacees par tokens, hashtags conserves sans `#`, nombres normalises, ponctuation et caracteres non ASCII retires.

Distribution cible: negative {class_dist['negative']:.1f} %, neutral {class_dist['neutral']:.1f} %, positive {class_dist['positive']:.1f} %.

## Resultats

Le meilleur modele au test selon F1-macro est **{best['model']}** avec F1-macro = **{best['macro_f1']:.3f}** et accuracy = **{best['accuracy']:.3f}**.

Modele retenu pour l'application Streamlit: **{deploy_model}**. Il est robuste, rapide a charger et interpretable via les poids TF-IDF.

Voir les tableaux:

- `outputs/tables/comparaison_modeles.csv`
- `outputs/tables/analyse_erreurs_echantillon.csv`
- `outputs/tables/top_tfidf_terms.csv`
- `outputs/tables/word2vec_voisins.csv`

Voir les figures dans `outputs/figures/`.

## Lecture metier

La classe `neutral` reste la plus fragile: elle porte souvent des demandes factuelles, des tweets ambigus ou des messages qui mentionnent un probleme sans emotion explicite. Le F1-macro est donc plus fiable que l'accuracy pour juger la valeur du systeme.

Pour une mise en production, le modele doit etre reentraine sur des tweets recents, surveille par compagnie, et complete par une priorisation metier qui combine sentiment negatif, retweets et mots lies aux incidents.
"""
    (REPORTS_DIR / "rapport_synthese.md").write_text(content, encoding="utf-8")


def save_metadata(results_df: pd.DataFrame, deploy_model_path: Path) -> None:
    metadata = {
        "classes": CLASSES,
        "random_state": RANDOM_STATE,
        "text_cleaning": "lowercase, URL/user placeholders, hashtag text kept, numbers normalized, punctuation removed",
        "primary_metric": "macro_f1",
        "selected_model_path": str(deploy_model_path),
        "results": results_df.to_dict(orient="records"),
    }
    with (ARTIFACTS_DIR / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main() -> None:
    set_global_seed()
    ensure_dirs()

    df = load_dataset()
    df.to_csv(PROCESSED_DIR / "tweets_clean.csv", index=False)
    generate_eda(df)
    splits = create_splits(df)
    y_train, _, _ = label_arrays(splits)
    weights = class_weights(y_train)

    results: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}

    logistic_model, logistic_result, logistic_pred = train_logistic_tfidf(splits)
    results.append(logistic_result)
    predictions["TF-IDF + Logistic Regression"] = logistic_pred

    _, ann_result, ann_pred = train_ann_tfidf(splits, weights)
    results.append(ann_result)
    predictions["TF-IDF + ANN"] = ann_pred

    deep_results, deep_predictions = train_deep_models(splits, weights)
    results.extend(deep_results)
    predictions.update(deep_predictions)

    # DistilBERT (frozen) features + logistic regression. Optional: skipped with a
    # clear message if transformers / torch are not installed, so the core
    # pipeline always completes.
    try:
        from src.bert_distilbert import train_distilbert_features

        bert_result, bert_pred = train_distilbert_features(splits)
        results.append(bert_result)
        predictions[bert_result["model"]] = bert_pred
    except Exception as exc:  # pragma: no cover - optional dependency / runtime
        print(f"[DistilBERT] skipped feature extraction: {exc}")

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    results_df.to_csv(TABLES_DIR / "comparaison_modeles.csv", index=False)
    save_error_analysis(splits.test, predictions)

    deploy_model_path = ARTIFACTS_DIR / "selected_tfidf_model.joblib"
    joblib.dump(logistic_model, deploy_model_path)
    deploy_model_name = "TF-IDF + Logistic Regression"
    write_model_card(results_df, df, splits, deploy_model_name)
    save_metadata(results_df, deploy_model_path)

    print("\nModel comparison:")
    print(results_df[["model", "accuracy", "macro_f1", "weighted_f1"]].to_string(index=False))
    print(f"\nSaved deployable model: {deploy_model_path}")
    print(f"Report: {REPORTS_DIR / 'rapport_synthese.md'}")


if __name__ == "__main__":
    main()
