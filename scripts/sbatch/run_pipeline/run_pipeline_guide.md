# Anleitung zur Ausführung der gesamten Pipeline

Diese Anleitung beschreibt alle Schritte, um die Daten-, Trainings- und Evaluierungs-Pipeline der Masterarbeit von Grund auf neu aufzusetzen, die Daten zu crawlen, vorzuverarbeiten und die Modelle (Klassifikation, MixUp-Regression, SFT, DPO) sowie sämtliche Experimente auf einem GPU-Server (HPC Slurm-Cluster) auszuführen und zu evaluieren.

---

## 0. Vorbereitung & Synchronisation auf den HPC-Server (rsync)

Um die Pipeline auf dem Server auszuführen, müssen **die Skripte, Konfigurationen und externen/lokalen Rohdaten** übertragen werden. Nicht benötigt werden:
- Lokale Umgebungen & Bytecode (`.venv`, `__pycache__`, `*.pyc`, `download`)
- Metadaten & Notizen (`.git`, `.obsidian`, `.DS_Store`, `research`)
- Lokale Web-App & LaTeX-Arbeit (`web`, `templates`, `thesis`)
- Bereits existierende Ergebnisse & Logs (`results`, `*.log`, `notebooks`)
- Generierte Zwischendaten (`data/lebenshilfe`, `data/vocabs`, `data/corpus/3_*`, `4_*`, `5_*`), da diese von den SBATCH-Skripten Schritt für Schritt selbst erzeugt werden.

> **Wichtig:** Externe Benchmark-Daten wie `data/analysis/textcomplexityde/` sowie die lokalen Lebenshilfe-Quelldokumente in `data/texts_lebenshilfe/` sind Rohdaten und müssen auf den Server synchronisiert werden.

---

### 0.1 Modularer Sync (Schritt für Schritt)

#### Schritt 1: Pipeline-Skripte, Quellcode & Abhängigkeiten
```bash
# Skripte & Projektdateien übertragen
rsync -avz ./scripts/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/scripts/
rsync -avz ./requirements.txt fisc4884@hpc3.hs-flensburg.de:~/master-thesis/requirements.txt
```

#### Schritt 2: Lokale Rohdaten & externe Benchmarks
```bash
# Lebenshilfe-Rohdateien
rsync -avz ./data/texts_lebenshilfe/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/data/texts_lebenshilfe/

# TextComplexityDE Benchmark-Dateien (Ratings & Source)
rsync -avz ./data/analysis/textcomplexityde/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/data/analysis/textcomplexityde/
```

#### Schritt 3 (Optional): Web-Scraping überspringen & vorhandene Rohdaten mitnehmen
Falls das zeitaufwändige Web-Crawling übersprungen werden soll:
```bash
rsync -avz ./data/corpus/1_aligned_urls/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/data/corpus/1_aligned_urls/
rsync -avz ./data/corpus/2_raw_scraped/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/data/corpus/2_raw_scraped/
```

---

### 0.2 All-in-One Befehle (mit vollständigen Ausschlüssen)

#### All-in-One: Szenario A (Komplett neu crawlen)
```bash
rsync -avz \
  --exclude='.venv' \
  --exclude='download' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='.obsidian' \
  --exclude='.DS_Store' \
  --exclude='web' \
  --exclude='thesis' \
  --exclude='templates' \
  --exclude='research' \
  --exclude='notebooks' \
  --exclude='results' \
  --exclude='*.log' \
  --exclude='data/corpus' \
  --exclude='data/lebenshilfe' \
  --exclude='data/vocabs' \
  --exclude='data/analysis/corpus_master.*' \
  --exclude='data/analysis/rule_adherence_*' \
  --exclude='data/analysis/information_loss_*' \
  --exclude='data/analysis/similarity_extremes.json' \
  --exclude='data/analysis/mixup_test_split.csv' \
  ./ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/
```

---

### 0.3 Basis-Ordnerstruktur auf dem Server anlegen

Logge dich auf dem Server ein und erstelle die Zielverzeichnisse für die Daten- und Modellerzeugung sowie das strukturierte Logging:

```bash
ssh fisc4884@hpc3.hs-flensburg.de "mkdir -p ~/master-thesis/data/{corpus,lebenshilfe,analysis/textcomplexityde,vocabs,temperature_ladder_500,dpo,metric_weights_exp} ~/master-thesis/results/{models/{sft,dpo,decoder_only,temperature_ladder_500},logs/{run_pipeline,experiments},plots/{run_pipeline,experiments},evaluation}"
```

---

## 1. Setup & Installation (Python Environment)

Auf dem HPC-Server ausführen:

```bash
# 1. Virtuelle Umgebung erstellen und aktivieren
python3 -m venv .venv
source .venv/bin/activate

# 2. Abhängigkeiten installieren
pip install --upgrade pip
pip install -r requirements.txt

# 3. SpaCy-Sprachmodelle herunterladen (für Lemmatisierung, NER & Metriken)
python3 -m spacy download de_core_news_lg
python3 -m spacy download de_core_news_sm
```

---

## 2. Kanonische Standard-Pipeline (`scripts/sbatch/run_pipeline/`)

Die Standard-Pipeline besteht aus 13 sequentiell aufeinander aufbauenden Schritten:

```
[01/02 Web Scraping] & [03/04 Lebenshilfe] 
                    │
                    ▼
          [05 Corpus Master] 
                    │
    ┌───────────────┼──────────────────────────────┐
    ▼               ▼                              ▼
[06 10kGNAD DPO] [07/08 Klassifikatoren]  [09 MixUp-Regressor]
    │                                              │
    │                                              ▼
    │                                      [10 SFT Training]
    │                                              │
    └───────────────────────┬──────────────────────┘
                            ▼
              [11 DPO Paar-Generierung]
                            ▼
                   [12 DPO Training]
                            ▼
         [13 Finale Benchmark-Evaluierung (Lebenshilfe)]
```

### 2.1 Übersicht der SBATCH-Skripte

| Schritt | Skript | Ressource | Beschreibung |
|:---|:---|:---|:---|
| **01** | `01_crawl_url_alignment.sh` | CPU (4 Cores, 16GB) | URL-Alignment aller 12 Webquellen |
| **02** | `02_crawl_content_extraction.sh` | CPU (4 Cores, 16GB) | HTML-Text-Extraktion und Vorbereinigung |
| **03** | `03_create_lebenshilfe_dataset.sh` | CPU (4 Cores, 16GB) | Lokale Lebenshilfe-Rohdateien einlesen |
| **04** | `04_clean_lebenshilfe.sh` | CPU (4 Cores, 16GB) | Bereinigung des Lebenshilfe-Datensatzes |
| **05** | `05_build_corpus_master.sh` | GPU (MIG 24GB) | Filterung, Deduplizierung, CSV/JSON Master-Korpus |
| **06** | `06_prepare_10kgnad_dpo_corpus.sh` | CPU (4 Cores, 16GB) | 10kGNAD Alltagssprache für DPO aufbereiten |
| **07** | `07_train_sentence_classifier.sh` | GPU (MIG 24GB) | Satz-Klassifikator (BiLSTM) trainieren |
| **08** | `08_train_article_classifier.sh` | GPU (MIG 24GB) | Artikel-Klassifikator (BiLSTM) trainieren |
| **09** | `09_train_mixup_regressor.sh` | GPU (MIG 24GB) | MixUp Style-Score Regressor trainieren |
| **10** | `10_train_sft.sh` | GPU (MIG 24GB) | mBART-50 SFT Training auf `corpus_master.json` |
| **11** | `11_generate_dpo_dataset.sh` | GPU (MIG 24GB) | DPO-Paare auf ungesehenem 10kGNAD generieren |
| **12** | `12_train_dpo.sh` | GPU (MIG 24GB) | LoRA DPO Training (mBART-50) |
| **13** | `13_evaluate_pipeline.sh` | GPU (MIG 24GB) | End-to-End Evaluierung auf dem Lebenshilfe-Benchmark |

### 2.2 Ausführung aller Schritte als Slurm-Job-Kette

Um die gesamte Pipeline automatisiert mit Job-Abhängigkeiten (`--dependency=afterok`) zu starten:

```bash
bash scripts/sbatch/run_pipeline/run_all_pipeline.sh
```

### 2.3 Modulare Ausführung nach Themenbereichen (Teil für Teil)

Falls nur bestimmte Phasen oder Komponenten neu gerechnet oder getestet werden sollen:

| Themenbereich | Skript | Enthaltene Einzelschritte |
|:---|:---|:---|
| **1. Scraping & Crawling** | `bash scripts/sbatch/run_pipeline/run_01_scraping.sh` | 01 $\rightarrow$ 02 |
| **2. Lebenshilfe Vorbereitung** | `bash scripts/sbatch/run_pipeline/run_02_lebenshilfe_prep.sh` | 03 $\rightarrow$ 04 |
| **3. Korpus-Erstellung** | `bash scripts/sbatch/run_pipeline/run_03_corpus_building.sh` | 05, 06 |
| **4. Reward- & Metrik-Modelle** | `bash scripts/sbatch/run_pipeline/run_04_reward_models.sh` | 07, 08, 09 (parallel) |
| **5. SFT-Training** | `bash scripts/sbatch/run_pipeline/run_05_sft_training.sh` | 10 |
| **6. DPO-Pipeline** | `bash scripts/sbatch/run_pipeline/run_06_dpo_pipeline.sh` | 11 $\rightarrow$ 12 |
| **7. Pipeline-Evaluierung** | `bash scripts/sbatch/run_pipeline/run_07_evaluation.sh` | 13 |

---

## 3. Experimente & Ablationen (`scripts/sbatch/experiments/`)

Alle wissenschaftlichen Experimente und Ablationsstudien der Masterarbeit sind modular in `scripts/sbatch/experiments/` organisiert und besitzen jeweils eigene Runner- und Evaluations-Skripte:

### 3.1 Modell- & Sequenzlängen-Ablationen

1. **Temperature Ladder 500 Tokens (`scripts/sbatch/experiments/temperature_ladder_500/`):**
   - DPO-Präferenzdatengenerierung via Temperature Ladder Sampling (500 Tokens) mit Gewichtungskonfigurationen ($w_{1.0}/w_{0.0}$ und $w_{0.5}/w_{0.5}$) sowie LoRA DPO-Training.
   - Master-Runner: `bash scripts/sbatch/experiments/temperature_ladder_500/run_all.sh` (oder lokal via `run_standalone.sh`)

2. **Token Length Scaling (256, 512, 1024 Tokens) (`scripts/sbatch/experiments/token_length/`):**
   - Untersuchung des Einflusses der Sequenzlänge auf Reward-Modell, SFT-Übersetzung und DPO.
   - Master-Runner: `bash scripts/sbatch/experiments/token_length/run_all_token_experiments.sh`

3. **Token Length mit Jina Embeddings (`scripts/sbatch/experiments/token_length_jina/`):**
   - Vergleich der semantischen Konsistenzmetrik unter Verwendung von Jina-Embeddings (`jinaai/jina-embeddings-v2-base-de`).
   - Master-Runner: `bash scripts/sbatch/experiments/token_length_jina/run_all_token_jina_experiments.sh`

4. **Length Bias Analyse (`scripts/sbatch/experiments/length_bias/`):**
   - Prüfung von Reward-Modell und DPO-Präferenzen auf systematische Längenverzerrungen (Length Bias).
   - Starten: `sbatch scripts/sbatch/experiments/length_bias/1_check_length_bias.sh`

### 3.2 Architektur-Vergleiche (Encoder-Decoder vs. Decoder-Only)

5. **Decoder-Only Pipeline (Qwen2.5-1.5B-Instruct) (`scripts/sbatch/experiments/decoder_only/`):**
   - Komplette Kette für autoregressive LLMs: SFT-Training via SFTTrainer, DPO-Paar-Generierung mit Prompt-Templates, DPOTrainer mit Shared Reference Model und Evaluierung.
   - Master-Runner: `bash scripts/sbatch/experiments/decoder_only/run_all_decoder_only.sh`

6. **RNN Baseline vs. BiLSTM (`scripts/sbatch/experiments/rnn_baseline/`):**
   - Vergleich des BiLSTM-Regressors mit Vanilla Elman-RNN und Unidirektionalem LSTM.
   - Starten: `sbatch scripts/sbatch/experiments/rnn_baseline/1_train_rnn_baseline.sh` $\rightarrow$ `sbatch scripts/sbatch/experiments/rnn_baseline/2_evaluate_rnn_baseline.sh`

### 3.3 Skalierungs- & Datensatz-Experimente

7. **SFT Data Scaling (`scripts/sbatch/experiments/sft_scaling/`):**
   - Skalierungskurven des mBART-50 SFT Trainings bei Datensatzgrößen von 250, 500, 1000, 1500 und 2000 Artikeln.
   - Master-Runner: `bash scripts/sbatch/experiments/sft_scaling/run_all_sft_scaling.sh`

8. **Data Scaling MixUp (`scripts/sbatch/experiments/data_scaling/`):**
   - 2D-Grid-Skalierung des MixUp-Regressors über Mischungsanzahlen (1k–20k) und Artikelanzahlen.
   - Master-Runner: `bash scripts/sbatch/experiments/data_scaling/run_all_data_scaling.sh`

9. **Synthetischer Regressor (`scripts/sbatch/experiments/synthetic_regressor/`):**
   - Generierung kontinuierlicher Zwischenstufen via LLM API (`FlensGen-GPT-OSS-120B`) und Training eines synthetischen Reward-Modells.
   - Master-Runner: `bash scripts/sbatch/experiments/synthetic_regressor/run_all_synthetic_pipeline.sh`

10. **Glossar-Extraktion & Anreicherung (`scripts/sbatch/experiments/glossary/`):**
    - Extraktion von Begriffserklärungen und Anreicherung des Korpus mit Glossar-Annotationen.
    - Starten: `sbatch scripts/sbatch/experiments/glossary/1_build_glossary.sh` $\rightarrow$ `sbatch scripts/sbatch/experiments/glossary/2_enrich_glossary.sh`

### 3.4 DPO- & Reward-Hyperparameter

11. **Metric Weights Grid ($w_{\text{style}}$ vs. $w_{\text{sem}}$) (`scripts/sbatch/experiments/metric_weights/`):**
    - Grid-Search über Belohnungsgewichte ($0.5/0.5$, $0.7/0.3$, $1.0/0.0$) bei der DPO-Paar-Auswahl.
    - Master-Runner: `bash scripts/sbatch/experiments/metric_weights/run_all_metric_weights_experiments.sh`

12. **Loss Aggregation (mean vs. sum) (`scripts/sbatch/experiments/loss_aggregation/`):**
    - Untersuchung von `mean`- vs. `sum`-Aggregation in der DPO-Verlustfunktion.
    - Master-Runner: `bash scripts/sbatch/experiments/loss_aggregation/run_all_loss_aggregation_experiments.sh`

### 3.5 Evaluation & Externe Validierung

13. **Master 5-Wege-Benchmark (`scripts/sbatch/experiments/benchmark/`):**
    - Vergleichender Gesamt-Benchmark über alle trainierten Modellfamilien (mBART-50 Baseline/SFT/DPO vs. Qwen2.5 Baseline/SFT/DPO) auf dem Lebenshilfe-Testset.
    - Starten: `sbatch scripts/sbatch/experiments/benchmark/1_run_all_models_benchmark.sh`

14. **Quantitative Regeltreue / Rule Adherence (`scripts/sbatch/experiments/rule_adherence/`):**
    - Quantitative Messung der Einhaltung formaler Leichte-Sprache-Regeln (Satzlänge, Silbenzahl, Passivkonstruktionen, Fremdwörter, Genitive).
    - Starten: `sbatch scripts/sbatch/experiments/rule_adherence/1_measure_rule_adherence.sh`

15. **Faktentreue & Halluzinationserkennung (`scripts/sbatch/experiments/factuality_metric/`):**
    - Benchmark zur Faktenkonsistenz (Vergleich SBERT vs. NLI vs. NER-Overlap vs. Number-Consistency) mit ROC-AUC-Evaluation.
    - Starten: `sbatch scripts/sbatch/experiments/factuality_metric/1_run_factuality_metric_experiment.sh`

16. **TextComplexityDE Validierung (`scripts/sbatch/experiments/textcomplexityde/`):**
    - Externe Validierung des MixUp-Style-Score-Modells auf dem standardisierten TextComplexityDE-Datensatz.
    - Starten: `sbatch scripts/sbatch/experiments/textcomplexityde/1_evaluate_textcomplexityde.sh`

---

## 4. Visualisierungen & Diagramme (`scripts/visualization/`)

Nach Abschluss der Trainings- und Evaluierungsläufe können die Abbildungen und Diagramme für die Masterarbeit mit den Skripten in `scripts/visualization/` generiert werden:

```bash
# 1. Diagramme zur Regeltreue (Radar-Plots, Boxplots, Balkenvergleiche)
python scripts/visualization/visualize_rule_adherence.py

# 2. Vergleichsplots der Übersetzungsmodelle (BLEU, SARI, Style-Score)
python scripts/visualization/generate_comparison_plots.py

# 3. Lesbarkeits- und Korpusanalyse-Plots
python scripts/visualization/visualize_readability.py
python scripts/visualization/visualize_analysis.py
python scripts/visualization/visualize_ttr.py

# 4. Zusammenfassender Review-Report
python scripts/visualization/generate_review_report.py
```
