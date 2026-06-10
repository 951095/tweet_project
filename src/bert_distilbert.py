"""DistilBERT representations for the airline sentiment project.

Two complementary approaches, both required by the assignment ("extraction de
features et/ou fine-tuning"):

1. Feature extraction (fast, CPU-friendly): a frozen pre-trained DistilBERT
   produces mean-pooled sentence embeddings that feed a logistic regression.
   This is wired into the main pipeline.

2. Fine-tuning (heavier): the whole DistilBERT is trained end-to-end via a
   classification head. Run it on demand with
   ``python -m src.bert_distilbert --finetune`` (slow on CPU).

The module reuses the evaluation / plotting helpers from ``project_pipeline`` so
that DistilBERT rows are produced exactly like every other model and land in the
same comparison table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.project_pipeline import (
    ARTIFACTS_DIR,
    CLASSES,
    DatasetSplits,
    RANDOM_STATE,
    TABLES_DIR,
    create_splits,
    ensure_dirs,
    evaluate_predictions,
    label_arrays,
    load_dataset,
    save_confusion_matrix,
    set_global_seed,
)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64
EMBED_BATCH = 32

FEATURE_MODEL_NAME = "DistilBERT (features) + LogReg"
FINETUNE_MODEL_NAME = "DistilBERT (fine-tuned)"


def _load_encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    return tokenizer, model


def embed_texts(texts: list[str], tokenizer=None, model=None) -> np.ndarray:
    """Mean-pooled DistilBERT embeddings (masked average of the last layer)."""
    import torch

    if tokenizer is None or model is None:
        tokenizer, model = _load_encoder()

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors.append(pooled.cpu().numpy().astype("float32"))
    return np.vstack(vectors)


def _cached_embeddings(name: str, texts: list[str], tokenizer, model) -> np.ndarray:
    path = ARTIFACTS_DIR / f"distilbert_emb_{name}.npy"
    if path.exists():
        cached = np.load(path)
        if cached.shape[0] == len(texts):
            return cached
    embeddings = embed_texts(texts, tokenizer, model)
    np.save(path, embeddings)
    return embeddings


def train_distilbert_features(splits: DatasetSplits) -> tuple[dict[str, object], np.ndarray]:
    """Frozen DistilBERT embeddings + logistic regression head."""
    from sklearn.linear_model import LogisticRegression

    print(f"[DistilBERT] extracting features with {MODEL_NAME} ...")
    tokenizer, model = _load_encoder()
    x_train = _cached_embeddings("train", splits.train["clean_text"].tolist(), tokenizer, model)
    x_test = _cached_embeddings("test", splits.test["clean_text"].tolist(), tokenizer, model)

    y_train, _, y_test = label_arrays(splits)
    clf = LogisticRegression(
        max_iter=2_000,
        class_weight="balanced",
        C=1.0,
        random_state=RANDOM_STATE,
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    result = evaluate_predictions(FEATURE_MODEL_NAME, y_test, y_pred)
    save_confusion_matrix(FEATURE_MODEL_NAME, y_test, y_pred)

    import joblib

    joblib.dump(clf, ARTIFACTS_DIR / "distilbert_features_logreg.joblib")
    return result, y_pred


def train_distilbert_finetune(splits: DatasetSplits, epochs: int = 3, batch_size: int = 16) -> tuple[dict[str, object], np.ndarray]:
    """End-to-end fine-tuning of DistilBERT with a classification head (slow on CPU)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    set_global_seed()
    y_train, _, y_test = label_arrays(splits)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(CLASSES))

    def encode(texts: list[str]):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    enc_train = encode(splits.train["clean_text"].tolist())
    enc_test = encode(splits.test["clean_text"].tolist())

    train_ds = TensorDataset(enc_train["input_ids"], enc_train["attention_mask"], torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Class weights to handle the imbalance (mirrors the rest of the pipeline).
    counts = np.bincount(y_train, minlength=len(CLASSES)).astype("float32")
    weights = torch.tensor(len(y_train) / (len(CLASSES) * np.maximum(counts, 1.0)), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(epochs):
        running = 0.0
        for step, (input_ids, attention_mask, labels) in enumerate(loader):
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss)
            if step % 50 == 0:
                print(f"[finetune] epoch {epoch + 1}/{epochs} step {step}/{len(loader)} loss {loss:.4f}")
        print(f"[finetune] epoch {epoch + 1} mean loss {running / max(1, len(loader)):.4f}")

    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        ids, masks = enc_test["input_ids"], enc_test["attention_mask"]
        for start in range(0, ids.shape[0], 64):
            logits = model(input_ids=ids[start : start + 64], attention_mask=masks[start : start + 64]).logits
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    y_pred = np.concatenate(preds)

    result = evaluate_predictions(FINETUNE_MODEL_NAME, y_test, y_pred)
    save_confusion_matrix(FINETUNE_MODEL_NAME, y_test, y_pred)
    model.save_pretrained(ARTIFACTS_DIR / "distilbert_finetuned")
    tokenizer.save_pretrained(ARTIFACTS_DIR / "distilbert_finetuned")
    return result, y_pred


def _merge_into_comparison(result: dict[str, object]) -> None:
    """Add/replace a DistilBERT row in the existing comparison table and re-sort."""
    path = TABLES_DIR / "comparaison_modeles.csv"
    if path.exists():
        table = pd.read_csv(path)
        table = table[table["model"] != result["model"]]
        table = pd.concat([table, pd.DataFrame([result])], ignore_index=True)
    else:
        table = pd.DataFrame([result])
    table = table.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    table.to_csv(path, index=False)
    print("\nUpdated comparison table:")
    print(table[["model", "accuracy", "macro_f1", "weighted_f1"]].to_string(index=False))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="DistilBERT models for airline sentiment.")
    parser.add_argument("--finetune", action="store_true", help="run end-to-end fine-tuning (slow on CPU)")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    set_global_seed()
    ensure_dirs()
    df = load_dataset()
    splits = create_splits(df)

    if args.finetune:
        result, _ = train_distilbert_finetune(splits, epochs=args.epochs)
    else:
        result, _ = train_distilbert_features(splits)

    _merge_into_comparison(result)
    print(f"\nDone: {result['model']} -> macro_f1 = {result['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
