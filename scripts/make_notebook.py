from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK_PATH = Path("notebooks/Projet_NLP_Sentiment_Tweets_Aeriens.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            "# Projet NLP - Sentiment des tweets aeriens\n\n"
            "Objectif: predire le sentiment (`negative`, `neutral`, `positive`) de tweets adresses "
            "a six compagnies aeriennes, puis fournir une demonstration deployable."
        ),
        md(
            "## 0. Cadrage business\n\n"
            "Problematique: classer automatiquement les tweets pour prioriser les reponses du service client, "
            "suivre la satisfaction par compagnie et detecter les signaux de crise.\n\n"
            "Questions business:\n\n"
            "1. Quelle compagnie a la plus forte part de tweets negatifs ?\n"
            "2. Quels motifs expliquent les tweets negatifs ?\n"
            "3. Quel modele offre le meilleur compromis F1-macro / cout de calcul / deployabilite ?\n\n"
            "Metrique principale: F1-macro, car la classe negative domine le dataset."
        ),
        code(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image\n\n"
            "from src.project_pipeline import load_dataset, create_splits, clean_tweet\n\n"
            "df = load_dataset()\n"
            "display(df.head())\n"
            "display(df['airline_sentiment'].value_counts(normalize=True).rename('part'))"
        ),
        md("## 1. Exploration et preparation\n\nLe pipeline nettoie les tweets et cree un split stratifie fige."),
        code(
            "splits = create_splits(df)\n"
            "print(len(splits.train), len(splits.validation), len(splits.test))\n"
            "display(splits.train[['airline_sentiment', 'text', 'clean_text']].head())"
        ),
        md("Les figures d'EDA sont generees par `python -m src.project_pipeline`."),
        code(
            "for path in sorted(Path('outputs/figures').glob('0*.png')):\n"
            "    print(path)\n"
            "    display(Image(filename=str(path)))"
        ),
        md(
            "## 2. Representations\n\n"
            "- TF-IDF: baseline explicable avec n-grammes.\n"
            "- Word2Vec-style: skip-gram entraine sur le corpus, reutilise comme initialisation des CNN/LSTM/Attention.\n"
            "- DistilBERT: embeddings contextuels pre-entraines (`distilbert-base-uncased`), "
            "utilises en extraction de features (modele gele + regression logistique) et en fine-tuning bout-en-bout.\n\n"
            "Les voisins semantiques sont sauvegardes dans `outputs/tables/word2vec_voisins.csv`."
        ),
        code(
            "neighbors = Path('outputs/tables/word2vec_voisins.csv')\n"
            "if neighbors.exists():\n"
            "    display(pd.read_csv(neighbors).head(20))\n"
            "else:\n"
            "    print('Lancez python -m src.project_pipeline pour generer ce tableau.')"
        ),
        md(
            "## 3. Modelisation\n\n"
            "Modeles entraines: regression logistique TF-IDF, ANN TF-IDF, CNN Word2Vec, BiLSTM Word2Vec, "
            "BiLSTM avec attention, et DistilBERT (features gelees + LogReg, ou fine-tuning).\n\n"
            "La partie DistilBERT se lance separement (elle necessite `torch` et `transformers`):\n\n"
            "```bash\n"
            "python -m src.bert_distilbert              # extraction de features (rapide sur CPU)\n"
            "python -m src.bert_distilbert --finetune   # fine-tuning bout-en-bout (lent sur CPU)\n"
            "```\n\n"
            "Les resultats sont fusionnes automatiquement dans `outputs/tables/comparaison_modeles.csv`."
        ),
        code(
            "results_path = Path('outputs/tables/comparaison_modeles.csv')\n"
            "if results_path.exists():\n"
            "    results = pd.read_csv(results_path)\n"
            "    display(results[['model', 'accuracy', 'macro_f1', 'weighted_f1']])\n"
            "else:\n"
            "    print('Lancez python -m src.project_pipeline pour entrainer les modeles.')"
        ),
        md("## 4. Evaluation et analyse d'erreurs"),
        code(
            "for path in sorted(Path('outputs/figures').glob('confusion_*.png')):\n"
            "    print(path.name)\n"
            "    display(Image(filename=str(path)))\n\n"
            "errors = Path('outputs/tables/analyse_erreurs_neutre.csv')\n"
            "if errors.exists():\n"
            "    display(pd.read_csv(errors).head(10))"
        ),
        md("## 5. Interpretabilite"),
        code(
            "for path in ['outputs/figures/interpretabilite_top_tfidf_terms.png', "
            "'outputs/figures/interpretabilite_attention_exemple.png']:\n"
            "    p = Path(path)\n"
            "    if p.exists():\n"
            "        print(p.name)\n"
            "        display(Image(filename=str(p)))"
        ),
        md(
            "## 6. Deploiement\n\n"
            "L'application se lance depuis un terminal avec:\n\n"
            "```bash\n"
            "streamlit run app/streamlit_app.py\n"
            "```"
        ),
    ]
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
