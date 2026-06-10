# Projet NLP - Sentiment des tweets aeriens

## Problematique business

Comment classer automatiquement les tweets adresses aux compagnies aeriennes en `negative`, `neutral` ou `positive` afin de prioriser les reponses du service client, suivre la satisfaction par compagnie et detecter rapidement les pics d'insatisfaction ?

Questions mesurables:

1. Quelle compagnie concentre la plus forte part de tweets negatifs ?
2. Quels motifs expliquent principalement l'insatisfaction ?
3. Quel compromis modele / cout de calcul donne le meilleur F1-macro pour une demonstration deployable ?

## Protocole

- Donnees: 14,640 tweets, split stratifie fige en train (10,248), validation (2,196) et test (2,196).
- Metrique principale: F1-macro, car les classes sont desequilibrees.
- Metriques secondaires: accuracy, F1 par classe, matrices de confusion et analyse d'erreurs.
- Desequilibre: `class_weight=balanced` pour les modeles supervises et lecture prioritaire du F1-macro.
- Nettoyage: minuscules, URLs et mentions remplacees par tokens, hashtags conserves sans `#`, nombres normalises, ponctuation et caracteres non ASCII retires.

Distribution cible: negative 62.7 %, neutral 21.2 %, positive 16.1 %.

## Resultats

Le meilleur modele au test selon F1-macro est **TF-IDF + Logistic Regression** avec F1-macro = **0.748** et accuracy = **0.798**.

Modele retenu pour l'application Streamlit: **TF-IDF + Logistic Regression**. Il est robuste, rapide a charger et interpretable via les poids TF-IDF.

Voir les tableaux:

- `outputs/tables/comparaison_modeles.csv`
- `outputs/tables/analyse_erreurs_echantillon.csv`
- `outputs/tables/top_tfidf_terms.csv`
- `outputs/tables/word2vec_voisins.csv`

Voir les figures dans `outputs/figures/`.

## Lecture metier

La classe `neutral` reste la plus fragile: elle porte souvent des demandes factuelles, des tweets ambigus ou des messages qui mentionnent un probleme sans emotion explicite. Le F1-macro est donc plus fiable que l'accuracy pour juger la valeur du systeme.

Pour une mise en production, le modele doit etre reentraine sur des tweets recents, surveille par compagnie, et complete par une priorisation metier qui combine sentiment negatif, retweets et mots lies aux incidents.
