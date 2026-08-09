# Master Thesis: Skripte-Verzeichnis (`scripts/`)

Dieses Verzeichnis enthält alle Python-Skripte für die Datenbeschaffung, Bereinigung, Evaluierung, Modellierung und Visualisierung im Rahmen der Masterarbeit zur automatischen Übersetzung in Leichte Sprache.

Der Ordner wurde strukturiert, um die Reproduzierbarkeit der Ergebnisse zu gewährleisten und den Pipeline-Verlauf logisch abzubilden.

---

## Verzeichnisstruktur & Pipeline-Schritte

Alle Skripte sind in fünf funktionale Unterordner unterteilt, die den logischen Ablauf des Projekts widerspiegeln:

```text
scripts/
├── data_collection/           # Schritt 1: Web-Scraping und URL-Alignment
│   ├── crawl_scraper/         # Stufe 1: Crawler zur URL-Findung und -Paarung
│   └── corpus_scrapers/       # Stufe 2: Scraper zum Extrahieren von Texten
├── preprocessing/             # Schritt 2: Filterung, Bereinigung und lokale Datensatzerstellung
│   ├── create_lebenshilfe_dataset.py
│   ├── clean_corpus.py
│   └── post_clean_corpus.py
├── evaluation/                # Schritt 3: Berechnung linguistischer & semantischer Metriken
│   ├── measure_information_loss.py
│   ├── info_loss_stats.py
│   ├── calculate_sbert_coverage.py
│   ├── count_total_tokens.py
│   ├── summarize_corpus.py
│   ├── measure_readability.py
│   └── measure_ttr.py
├── modeling/                  # Schritt 4: LLM-Generierung und Klassifikatorentraining/Evaluierung
│   ├── generate_synthetic_regression_steps.py
│   ├── evaluate_article_model.py
│   ├── evaluate_sentence_model.py
│   ├── check_length_bias.py
│   ├── test_llm.py
│   └── test_llm_openai.py
└── visualization/             # Schritt 5: Generierung von Abbildungen und Auswertungen
    ├── generate_comparison_plots.py
    ├── generate_review_report.py
    ├── visualize_analysis.py
    ├── visualize_readability.py
    └── visualize_ttr.py
```

---

## Ausführung der Pipeline

Führe alle Befehle aus dem **Hauptverzeichnis** (Repository-Root) aus.

### 1. Datenbeschaffung (`data_collection/`)
* **URL-Alignment (Stufe 1):** Sucht nach Paaren von Alltagssprache (AS) und Leichter Sprache (LS).
  ```bash
  .venv/bin/python scripts/data_collection/crawl_scraper/apotheken_scraper.py
  ```
* **Content-Extraction (Stufe 2):** Lädt HTML-Inhalte der gepaarten URLs herunter und extrahiert Fließtexte.
  ```bash
  .venv/bin/python scripts/data_collection/corpus_scrapers/apotheken_scraper.py
  ```

### 2. Vorverarbeitung & Bereinigung (`preprocessing/`)
* **Lokaler Datensatz (Lebenshilfe):**
  ```bash
  .venv/bin/python scripts/preprocessing/create_lebenshilfe_dataset.py
  ```
* **Korpus-Bereinigung:** Filterung basierend auf minimaler Länge und semantischer Ähnlichkeit ($0.60 \leq \text{Sim} \leq 0.99$).
  ```bash
  .venv/bin/python scripts/preprocessing/clean_corpus.py
  ```
* **Post-Processing:** Quellenspezifische Korrekturen (z.B. Mediopunkt-Entfernung).
  ```bash
  .venv/bin/python scripts/preprocessing/post_clean_corpus.py
  ```

### 3. Evaluierung & Metriken (`evaluation/`)
* **Semantische Ähnlichkeit & NER Recall:**
  ```bash
  .venv/bin/python scripts/evaluation/measure_information_loss.py --input_dir data/corpus/4_normalized_clean --output_csv data/analysis/information_loss_analysis_cleaned.csv
  ```
* **Lesbarkeits-Indizes (Flesch, Wiener Sachtextformel, LIX):**
  ```bash
  .venv/bin/python scripts/evaluation/measure_readability.py
  ```
* **Lexikalische Diversität (MATTR):**
  ```bash
  .venv/bin/python scripts/evaluation/measure_ttr.py
  ```
* **Zusammenfassende Statistiken:**
  ```bash
  .venv/bin/python scripts/evaluation/summarize_corpus.py
  ```

### 4. Modellierung & LLM-Synthese (`modeling/`)
* **Synthetische Datengenerierung via LLM-API:**
  ```bash
  .venv/bin/python scripts/modeling/generate_synthetic_regression_steps.py --url <API_URL> --token <TOKEN>
  ```
* **Modellevaluierung (LSTM-Klassifikatoren):**
  ```bash
  .venv/bin/python scripts/modeling/evaluate_article_model.py
  .venv/bin/python scripts/modeling/evaluate_sentence_model.py
  ```

### 5. Visualisierung (`visualization/`)
* **Grafiken zur Korpusanalyse:**
  ```bash
  .venv/bin/python scripts/visualization/visualize_analysis.py --plots all
  ```
* **Boxplots & Verteilungen zur Lesbarkeit & Diversität:**
  ```bash
  .venv/bin/python scripts/visualization/visualize_readability.py
  ```
