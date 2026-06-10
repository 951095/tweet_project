# Projet NLP — Sentiment des tweets aériens

Classification de sentiment (`negative`, `neutral`, `positive`) sur le dataset
**US Airline Sentiment** (~14 600 tweets adressés à six compagnies aériennes
américaines).

Le projet couvre toute la chaîne : exploration → nettoyage → représentations
(TF-IDF, Word2Vec maison, DistilBERT) → modèles classiques et profonds →
évaluation, interprétabilité et démonstration déployable.

---

## Sommaire

1. [Installation](#installation)
2. [Lancer le projet](#lancer-le-projet)
3. [Arborescence](#arborescence)
4. [Protocole expérimental](#protocole-expérimental)
5. [Modèles et résultats](#modèles-et-résultats)
6. [Livrables](#livrables)

---

## Installation

Python 3.10+ recommandé (testé sur 3.12).

```bash
# Optionnel mais conseillé : environnement isolé
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

pip install -r requirements.txt
```

> **Note DistilBERT** — `torch` et `transformers` ne sont nécessaires que pour la
> partie DistilBERT. S'ils ne sont pas installés, le pipeline principal s'exécute
> quand même et saute proprement cette étape (avec un message).

---

## Lancer le projet

Toutes les commandes se lancent **depuis la racine du projet** (`projet_tweet/`).

### 1. Pipeline complet (EDA + entraînement + évaluation)

Génère les splits, les figures, les modèles, les tableaux de résultats et le
rapport de synthèse.

```bash
python -m src.project_pipeline
```

Produit : `data/processed/`, `outputs/figures/`, `outputs/tables/`,
`artifacts/`, `reports/rapport_synthese.md`.

### 2. DistilBERT (optionnel)

```bash
# Extraction de features (DistilBERT gelé) + régression logistique — rapide sur CPU
python -m src.bert_distilbert

# Fine-tuning bout-en-bout — lent sur CPU
python -m src.bert_distilbert --finetune --epochs 3
```

La ligne DistilBERT est automatiquement ajoutée à
`outputs/tables/comparaison_modeles.csv`.

> **⚠️ Poids non versionnés** — le modèle fine-tuné
> `artifacts/distilbert_finetuned/model.safetensors` (~256 Mo) **n'est pas sur
> GitHub** (au-delà de la limite de 100 Mo, ignoré via `.gitignore`). Pour le
> reconstituer, relancer le fine-tuning ci-dessus (`--finetune`) ; les poids
> seront régénérés localement dans `artifacts/distilbert_finetuned/`. Le reste de
> la partie DistilBERT (extraction de features) ne nécessite pas ce fichier.

### 3. Application de démonstration (Streamlit)

Nécessite que le modèle déployable existe (étape 1 réalisée au moins une fois).

```bash
streamlit run app/streamlit_app.py
```

Saisis un tweet, l'app renvoie le sentiment prédit, la confiance, une
priorisation métier et la distribution de probabilités.

### 4. (Re)générer le notebook

Le notebook est construit par script pour rester synchronisé avec le pipeline :

```bash
python scripts/make_notebook.py
# puis l'ouvrir :
jupyter notebook notebooks/Projet_NLP_Sentiment_Tweets_Aeriens.ipynb
```

---

## Arborescence

```
projet_tweet/
├── README.md
├── requirements.txt
├── Projet_NLP_Sentiment_Tweets_Aeriens_v1.pdf   # énoncé du projet
│
├── tweets/                          # données brutes (entrée)
│   └── Tweets.csv                   # dataset US Airline Sentiment
│
├── src/                             # code source
│   ├── project_pipeline.py          # pipeline principal (EDA → modèles → rapport)
│   └── bert_distilbert.py           # DistilBERT (features + fine-tuning)
│
├── app/
│   └── streamlit_app.py             # démo interactive
│
├── scripts/
│   └── make_notebook.py             # génère le notebook
│
├── notebooks/
│   └── Projet_NLP_Sentiment_Tweets_Aeriens.ipynb
│
├── data/processed/                  # généré : splits stratifiés
│   ├── tweets_clean.csv
│   ├── train.csv / validation.csv / test.csv
│
├── artifacts/                       # généré : modèles + objets sérialisés
│   ├── selected_tfidf_model.joblib  # modèle servi par l'app Streamlit
│   ├── tfidf_logistic_regression.joblib
│   ├── ann_tfidf.keras + vectorizer
│   ├── cnn_word2vec.keras / bilstm_word2vec.keras
│   ├── word2vec_embeddings.npy / keras_tokenizer.joblib
│   └── metadata.json
│
├── outputs/                         # généré : résultats
│   ├── figures/                     # EDA, matrices de confusion, courbes, interprétabilité
│   ├── tables/                      # comparaison_modeles.csv, rapports, analyses d'erreurs
│   └── pipeline_run.log
│
└── reports/                         # généré : rapport de synthèse (md + pdf)
    └── rapport_synthese.md / .pdf
```

> Les dossiers `data/processed/`, `artifacts/`, `outputs/` et `reports/` sont
> **régénérés** par le pipeline ; ils sont versionnés ici pour fournir des
> résultats prêts à consulter. **Exception** : les poids fine-tunés
> `artifacts/distilbert_finetuned/model.safetensors` (~256 Mo) ne sont pas sur
> GitHub — voir [DistilBERT (optionnel)](#2-distilbert-optionnel) pour les
> régénérer.

---

## Protocole expérimental

- **Split** stratifié et figé (`random_state=42`) : 70 % train, 15 % validation,
  15 % test.
- **Anti-fuite** : déduplication sur le texte nettoyé *avant* le split (les
  retweets et réponses identiques ne traversent pas train/test).
- **Métrique principale** : F1-macro (classes déséquilibrées), complétée par
  l'accuracy, le F1 par classe, les matrices de confusion et une analyse
  d'erreurs.
- **Déséquilibre** : `class_weight="balanced"` à l'entraînement.
- **Nettoyage** : minuscules, URLs/mentions → tokens, hashtags conservés sans
  `#`, contractions développées, nombres normalisés, ponctuation et caractères
  non-ASCII retirés.

---

## Modèles et résultats

| Représentation | Modèle |
|---|---|
| TF-IDF (1-2 grammes) | Régression logistique *(modèle déployé)* |
| TF-IDF | ANN (MLP) |
| Word2Vec skip-gram (entraîné sur le corpus) | CNN |
| Word2Vec skip-gram | BiLSTM |
| Word2Vec skip-gram | BiLSTM + attention |
| DistilBERT | features + LogReg, ou fine-tuning *(optionnel)* |

Le classement complet est dans `outputs/tables/comparaison_modeles.csv`. À ce
jour, la **régression logistique TF-IDF** offre le meilleur compromis
F1-macro / coût / interprétabilité ; c'est le modèle servi par l'application.

---

## Livrables

- Pipeline complet : `src/project_pipeline.py`
- DistilBERT : `src/bert_distilbert.py`
- Application démo : `app/streamlit_app.py`
- Notebook : `notebooks/Projet_NLP_Sentiment_Tweets_Aeriens.ipynb`
- Rapport de synthèse : `reports/rapport_synthese.md` / `.pdf`
- Figures et tableaux : `outputs/figures/`, `outputs/tables/`
- Modèle déployable : `artifacts/selected_tfidf_model.joblib`
</content>
</invoke>

python -m src.bert_distilbert --finetune --epochs 3 