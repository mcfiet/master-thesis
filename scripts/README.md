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
├── preprocessing/             # Schritt 2: Filterung, Bereinigung, lokale Datensatzerstellung, Master-CSV & Synthetische Daten
│   ├── 0_create_lebenshilfe_dataset.py
│   ├── 1_filter_similarity.py
│   ├── 2_normalize_clean.py
│   ├── 2b_clean_lebenshilfe.py # Bereinigt Unterschriften, Prüfer-Hinweise und Metadaten-Rauschen aus Lebenshilfe
│   ├── 3_build_glossary.py
│   ├── 4_enrich_glossary.py
│   ├── 5_build_corpus_master.py # Erstellt die Master-CSV & JSON (Training- & Evaluierungsgrundlage)
│   └── 6_generate_synthetic_steps.py # Generiert Komplexitäts-Zwischenstufen via LLM API
├── evaluation/                # Schritt 3: Berechnung linguistischer & semantischer Metriken zur Korpusanalyse
│   ├── measure_information_loss.py
│   ├── info_loss_stats.py
│   ├── calculate_sbert_coverage.py
│   ├── count_total_tokens.py
│   ├── summarize_corpus.py
│   ├── measure_readability.py
│   └── measure_ttr.py
├── modeling/                  # Schritt 4: Modelltraining und Evaluierung
│   ├── 1_binary_train_sentence_model.py # Trainiert Satz-Klassifikator
│   ├── 2_binary_train_article_model.py  # Trainiert Artikel-Klassifikator
│   ├── 3_regression_train_mixup.py      # Trainiert MixUp-Regressor (Style-Score)
│   ├── 4_regression_train_synthetic.py  # Trainiert Synthetischen Regressor (Style-Score)
│   ├── 5_train_sft.py                   # Trainiert SFT-Modell (Übersetzung)
│   ├── 6_train_dpo.py                   # Trainiert DPO-Modell (Alignierung)
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

- **URL-Alignment (Stufe 1):** Sucht nach Paaren von Alltagssprache (AS) und Leichter Sprache (LS).
  ```bash
  .venv/bin/python scripts/data_collection/crawl_scraper/apotheken_scraper.py
  ```
- **Content-Extraction (Stufe 2):** Lädt HTML-Inhalte der gepaarten URLs herunter und extrahiert Fließtexte.
  ```bash
  .venv/bin/python scripts/data_collection/corpus_scrapers/apotheken_scraper.py
  ```

### 2. Vorverarbeitung & Datensatzerstellung (`preprocessing/`)

- **Lokaler Datensatz (Lebenshilfe) erstellen:** Verarbeitet lokale Dokumentdateien (`.docx`, `.rtf`, `.odt`) der Organisation _Lebenshilfe_.
  ```bash
  .venv/bin/python scripts/preprocessing/0_create_lebenshilfe_dataset.py
  ```
- **Lebenshilfe-Bereinigung:** Bereinigt den Lebenshilfe-Datensatz (entfernt Unterschriften, Prüfer-Hinweise und Metadaten-Rauschen).
  ```bash
  .venv/bin/python scripts/preprocessing/2b_clean_lebenshilfe.py \
      --input_file data/lebenshilfe/lebenshilfe_dataset.json \
      --output_file data/lebenshilfe/lebenshilfe_dataset_clean.json
  ```
- **Information Loss & Ähnlichkeits-Analyse:** Führt die Ähnlichkeitsanalyse aus, um die für die Filterung benötigte CSV-Datei zu erstellen.
  ```bash
  .venv/bin/python scripts/evaluation/measure_information_loss.py \
      --input_dir data/corpus/2_raw_scraped \
      --output_csv data/analysis/information_loss_analysis_cleaned.csv
  ```
- **Korpus-Bereinigung (Filterung):** Filterung basierend auf minimaler Länge und semantischer Ähnlichkeit ($0.60 \leq \text{Sim} \leq 0.99$) aus den zuvor berechneten Analysedaten.
  ```bash
  .venv/bin/python scripts/preprocessing/1_filter_similarity.py \
      --analysis_csv data/analysis/information_loss_analysis_cleaned.csv \
      --source_dir data/corpus/2_raw_scraped \
      --output_dir data/corpus/3_filtered_similarity \
      --sim_min 0.60 --sim_max 0.99 --min_ls_tokens 10
  ```
- **Post-Processing (Normalisierung):** Quellenspezifische Korrekturen (z.B. Mediopunkt-Entfernung).
  ```bash
  .venv/bin/python scripts/preprocessing/2_normalize_clean.py \
      --input_dir data/corpus/3_filtered_similarity \
      --output_dir data/corpus/4_normalized_clean
  ```
- **Glossar aufbauen:** Erstellt ein Vokabular aus dem Korpus via Hurraki API.
  ```bash
  .venv/bin/python scripts/preprocessing/3_build_glossary.py
  ```
- **Glossar-Augmentierung:** Reichert das Korpus mit Begriffserklärungen an.
  ```bash
  .venv/bin/python scripts/preprocessing/4_enrich_glossary.py
  ```
- **Master-CSV & JSON erstellen:** Berechnet alle Ähnlichkeits-, Lesbarkeits- und Diversitätsmetriken in einem einzigen Durchlauf. Es werden parallel eine `.csv` (für Datenanalysen) und eine `.json` (für robusteres Modelltraining) erstellt.
  ```bash
  .venv/bin/python scripts/preprocessing/5_build_corpus_master.py \
      --input_dir data/corpus/4_normalized_clean \
      --output_csv data/analysis/corpus_master.csv
  ```
- **Synthetische Datengenerierung via LLM-API:** Generiert Zwischenstufen (Standard: `0.25, 0.50, 0.75`) zwischen Alltagssprache ($0.0$) und Leichter Sprache ($1.0$) über ein OpenAI-kompatibles LLM API-Interface.

  ```bash
  # 1. Für den Lebenshilfe-Datensatz:
  .venv/bin/python scripts/preprocessing/6_generate_synthetic_steps.py \
      --input data/lebenshilfe/lebenshilfe_dataset_clean.json \
      --output data/lebenshilfe/lebenshilfe_dataset_with_steps.json \
      --url <API_URL> --token <TOKEN> --model "FlensGen-GPT-OSS-120B"

  # 2. Für das Hauptkorpus (aus dem Master-JSON):
  .venv/bin/python scripts/preprocessing/6_generate_synthetic_steps.py \
      --input data/analysis/corpus_master.json \
      --output data/corpus/corpus_master_with_steps.json \
      --url <API_URL> --token <TOKEN> --model "FlensGen-GPT-OSS-120B"
  ```

### 3. Evaluierung & Metriken (`evaluation/`)

- **Semantische Ähnlichkeit & NER Recall:**
  ```bash
  .venv/bin/python scripts/evaluation/measure_information_loss.py \
      --input_dir data/corpus/2_raw_scraped \
      --output_csv data/analysis/information_loss_analysis_cleaned.csv
  ```
- **Lesbarkeits-Indizes (Flesch, Wiener Sachtextformel, LIX):**
  ```bash
  .venv/bin/python scripts/evaluation/measure_readability.py
  ```
- **Lexikalische Diversität (MATTR):**
  ```bash
  .venv/bin/python scripts/evaluation/measure_ttr.py
  ```
- **Zusammenfassende Statistiken:**
  ```bash
  .venv/bin/python scripts/evaluation/summarize_corpus.py
  ```

### 4. Modellierung & Training (`modeling/`)

Trainiert Klassifikatoren, Regressoren sowie Übersetzungs- und DPO-Modelle. Alle Parameter müssen per CLI-Argument übergeben werden. Ausgaben werden live auf der Konsole ausgegeben und in `results/logs/` mitgeschrieben.

> [!NOTE]
> **Hardware-Ressourcen (CPU vs. GPU):**
> - **CPU-freundlich (GPU optional):** Die Metrik-Skripte (1, 2, 3 und 4) basieren auf kompakten BiLSTM-Netzen. Sie können problemlos auf der CPU trainiert werden.
> - **GPU zwingend erforderlich (GPU Mandatory):** Die Übersetzungs- und DPO-Skripte (5 und 6) trainieren bzw. tunen das große Transformer-Modell `mBART-large-50` (über 1 Mrd. Parameter). Ein Training auf der CPU führt aufgrund von Speichermangel und extrem langsamer Rechengeschwindigkeit zu Abbruch oder Nicht-Machbarkeit.

- **Satz-Klassifikator (BiLSTM):**
  ```bash
  .venv/bin/python scripts/modeling/1_binary_train_sentence_model.py \
      --csv_path data/analysis/corpus_master.csv \
      --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
      --batch_size 64 --embedding_dim 128 --epochs 20 --hidden_dim 128 \
      --lr 0.001 --max_seq_len 100 --max_sim 0.98 --min_sent_len 3 --min_sim 0.8
  ```
- **Artikel-Klassifikator (BiLSTM):**
  ```bash
  .venv/bin/python scripts/modeling/2_binary_train_article_model.py \
      --csv_path data/analysis/corpus_master.csv \
      --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
      --batch_size 32 --embedding_dim 128 --epochs 30 --hidden_dim 128 \
      --lr 0.001 --max_seq_len 512 --max_sim 0.98 --min_sim 0.8
  ```
- **MixUp-Regressor (Style-Score):**
  ```bash
  .venv/bin/python scripts/modeling/3_regression_train_mixup.py \
      --csv_path data/analysis/corpus_master.csv \
      --batch_size 64 --embedding_dim 128 --epochs 40 --hidden_dim 128 \
      --lr 0.001 --max_sim 0.98 --min_sim 0.8 --max_seq_len 256
  ```
- **Synthetischer Regressor (Style-Score):**
  ```bash
  .venv/bin/python scripts/modeling/4_regression_train_synthetic.py \
      --corpus_with_steps_path data/corpus/corpus_master_with_steps.json \
      --lh_with_steps_path data/lebenshilfe/lebenshilfe_dataset_with_steps.json \
      --model_save_path results/models/bilstm_synthetic_regression.pt \
      --vocab_save_path data/vocabs/synthetic_vocab.json \
      --epochs 15 --max_seq_len 256
  ```
- **Supervised Fine-Tuning (SFT Übersetzungsmodell):**
  ```bash
  .venv/bin/python scripts/modeling/5_train_sft.py \
      --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
      --corpus_path data/analysis/corpus_master.csv \
      --min_sim 0.70 --max_sim 0.98 --max_source_len 256 --max_target_len 256 \
      --model_name facebook/mbart-large-50 \
      --batch_size 8 --epochs 15 --lr 1e-5 --warmup_ratio 0.10 \
      --patience 5 --seed 42 --val_split 0.15 --output_dir results/models/seq2seq_sft \
      --reward_model_path results/models/bilstm_synthetic_regression.pt \
      --reward_vocab_path data/vocabs/synthetic_vocab.json
  ```
- **Direct Preference Optimization (DPO Ausrichtung):**
  ```bash
  .venv/bin/python scripts/modeling/6_train_dpo.py \
      --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
      --corpus_path data/analysis/corpus_master.csv \
      --output_dir results/models/seq2seq_dpo \
      --sft_model_dir results/models/seq2seq_sft \
      --reward_model_path results/models/bilstm_synthetic_regression.pt \
      --reward_vocab_path data/vocabs/synthetic_vocab.json \
      --min_sim 0.80 --max_sim 0.98 --w_style 0.5 --w_sem 0.5 \
      --max_source_len 256 --max_target_len 256 --model_name facebook/mbart-large-50
  ```

### 5. Visualisierung (`visualization/`)

- **Grafiken zur Korpusanalyse:**
  ```bash
  .venv/bin/python scripts/visualization/visualize_analysis.py --plots all
  ```
- **Boxplots & Verteilungen zur Lesbarkeit & Diversität:**
  ```bash
  .venv/bin/python scripts/visualization/visualize_readability.py
  ```
- **Type-Token-Ratio (TTR) visualisieren:**
  ```bash
  .venv/bin/python scripts/visualization/visualize_ttr.py
  ```
