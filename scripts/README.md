# Master Thesis: Skripte-Verzeichnis (`scripts/`)

Dieses Verzeichnis enthält alle Python-Skripte für die Datenbeschaffung, Bereinigung, Evaluierung, Modellierung, Experimente und Visualisierung im Rahmen der Masterarbeit zur automatischen Übersetzung in Leichte Sprache.

---

## Verzeichnisstruktur & Module

```text
scripts/
├── data/                      # Daten-Download & Aufbereitung externer Korpora (10kGNAD)
├── data_collection/           # Schritt 1: Web-Scraping und URL-Alignment
│   ├── crawl_scraper/         # Stufe 1: Crawler zur URL-Findung und -Paarung
│   └── corpus_scrapers/       # Stufe 2: Scraper zum Extrahieren von Texten
├── preprocessing/             # Schritt 2: Filterung, Bereinigung, lokale Datensätze, Master CSV/JSON
│   ├── build_corpus_master.py
│   ├── clean_lebenshilfe.py
│   └── create_lebenshilfe_dataset.py
├── modeling/                  # Schritt 3: Hauptmodellierung (Klassifikation, MixUp, SFT, DPO)
│   ├── binary_train_article_model.py
│   ├── binary_train_sentence_model.py
│   ├── generate_dpo_dataset.py
│   ├── regression_train_mixup.py
│   ├── train_dpo.py
│   └── train_sft.py
├── evaluation/                # Schritt 4: Metriken, Korpusanalysen & Modell-Benchmarks
│   ├── evaluate_dpo_ladder_model.py
│   ├── evaluate_article_model.py
│   ├── evaluate_sentence_model.py
│   ├── measure_information_loss.py
│   ├── measure_readability.py
│   ├── measure_ttr.py
│   └── summarize_corpus.py
├── experiments/               # Experimente & Ablationsstudien
│   ├── synthetic_regressor/   # LLM-Zwischenstufen & Synthetischer Regressor
│   ├── data_scaling/          # Daten-Skalierungsstudien
│   ├── sft_scaling/           # SFT-Epochen- & Daten-Skalierung
│   ├── glossary/              # Glossar-Augmentierung
│   └── run_factuality_metric_experiment.py
├── sbatch/                    # Slurm SBATCH Ausführungsskripte für HPC
│   ├── run_pipeline/          # Kanonische 13-stufige Standard-Pipeline
│   └── experiments/           # SBATCH-Skripte für alle Experimente
└── visualization/             # Abbildungen & Plots für die Thesis
```

---

## Haupt-Pipeline Ausführung

Die Haupt-Pipeline kann vollständig über das Master-Skript oder modular nach Themenbereichen gestartet werden:

```bash
# Vollständige Pipeline (alle 13 Schritte mit Slurm-Dependencies):
bash scripts/sbatch/run_pipeline/run_all_pipeline.sh

# Modular nach Themenbereichen (z.B. nur Reward-Modelle oder DPO-Stufe):
bash scripts/sbatch/run_pipeline/run_04_reward_models.sh
bash scripts/sbatch/run_pipeline/run_06_dpo_pipeline.sh
```

Details zu jedem Schritt und allen Themenbereich-Runnern siehe [`run_pipeline_guide.md`](../run_pipeline_guide.md) sowie [`scripts/sbatch/run_pipeline/README.md`](sbatch/run_pipeline/README.md).

