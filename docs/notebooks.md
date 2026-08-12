# Notebook-Referenz (Jupyter Notebooks)

Diese Dokumentation beschreibt die Jupyter Notebooks im Ordner `notebooks/`. Die Notebooks sind in zwei Hauptkategorien unterteilt: **Modelltraining** und **Modellevaluierung/Analyse (Research)**.

Alle älteren, alternativen oder experimentellen Notebook-Varianten wurden in den Ordner `notebooks/old/` verschoben.

---

## 1. Modelltraining (Jupyter Notebooks)

Diese Notebooks dienen dem eigentlichen Training der Modelle (Klassifikatoren, Regressoren, Übersetzung & DPO). Sie entsprechen 1-zu-1 den lauffähigen Python-Skripten unter `scripts/modeling/`.

### Satz- & Artikel-Klassifikation (`notebooks/metric/binary/`)
* **`1_train_sentence_model.ipynb`**: Trainiert den BiLSTM Satz-Klassifikator (AS vs. LS) auf Satzebene.
* **`2_train_article_model.ipynb`**: Trainiert den BiLSTM Artikel-Klassifikator (AS vs. LS) auf Artikelebene.

### MixUp Regression (`notebooks/metric/mixup/`)
* **`3b_mixup_hybrid_cyclic.ipynb`**: Trainiert den BiLSTM MixUp-Regressor auf kontinuierlichen Komplexitätsstufen (Hybrid-Dataloader mit zyklischer Lernrate).

### Synthetische Regression (`notebooks/metric/synthetic/`)
* **`1_synthetic_bilstm_regression.ipynb`**: Trainiert den BiLSTM Regressor auf den vom LLM synthetisch erzeugten Zwischenstufen (`0.25`, `0.50`, `0.75`).

### Übersetzung & DPO-Tuning (`notebooks/translation/`)
* **`2_sft.ipynb`**: Führt das Supervised Fine-Tuning (SFT) des mBART-Übersetzungsmodells auf den bereinigten Korpuspaaren durch. Speichert das Modell unter `results/models/2_sft.pt`.
* **`3_dpo.ipynb`**: Führt das Direct Preference Optimization (DPO) Tuning auf dem SFT-Modell unter Verwendung der Style-Reward-Funktion durch. Speichert unter `results/models/seq2seq_dpo`.

---

## 2. Modellevaluierung & Analyse (`notebooks/research/`)

Diese Notebooks werden ausschließlich für statistische Analysen, Visualisierungen und Performance-Vergleiche der trainierten Modelle und Daten verwendet.

### Daten-Analyse (`notebooks/research/data/`)
* **`corpus_diagnostics.ipynb`**: Diagnose und linguistische Analyse des Textkorpus.
* **`analyze_boilerplate_bias.ipynb`**: Untersucht den Einfluss von Boilerplate/Footer-Texten auf das Korpus.

### Metrik-Analyse (`notebooks/research/metric/`)
* **`check_length_bias.ipynb`**: Analysiert, ob die Klassifikatoren einen Längen-Bias aufweisen.
* **`check_metric_similarity.ipynb`**: Untersucht Ähnlichkeiten zwischen verschiedenen Metrik-Ansätzen.
* **`compare_mixup_vs_synthetic.ipynb`**: Vergleicht die Performance des MixUp-Regressors direkt mit dem Synthetischen Regressor auf dem Lebenshilfe-Set.
* **`4_mixup_model_evaluation.ipynb`**: Detaillierte Auswertung und Visualisierung (Scatterplots, Dichtediagramme) der trainierten MixUp-Modelle.

### Übersetzungs-Analyse (`notebooks/research/translation/`)
* **`compare_dpo_results.ipynb`**: Vergleicht die Ergebnisse (Kosinus-Ähnlichkeit, Style-Scores) verschiedener DPO-Modellläufe.
* **`compare_sft_vs_dpo_w10.ipynb`**: Direkter qualitativer und quantitativer Vergleich zwischen dem SFT-Modell und dem DPO-Modell.
