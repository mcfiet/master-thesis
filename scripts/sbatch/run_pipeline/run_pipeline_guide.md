# Anleitung zur Ausführung der gesamten Pipeline & Experimente

Diese Anleitung beschreibt alle Schritte, um die Daten-, Trainings- und Evaluierungs-Pipeline der Masterarbeit von Grund auf neu aufzusetzen, die Daten zu crawlen, vorzuverarbeiten und die Modelle (Klassifikation, MixUp-Regression, SFT, DPO) sowie sämtliche 17 Experimente auf einem GPU-Server (HPC Slurm-Cluster) automatisiert auszuführen und zu evaluieren.

---

## 0. Vorbereitung & Server-Setup

Um Probleme mit fehlenden Pfaden oder fehleranfälligen Ausschlusslisten (`--exclude`) zu vermeiden, wird zuerst die Ordnerstruktur auf dem Server erstellt und anschließend **ausschließlich das synchronisiert, was wirklich benötigt wird** (Whitelist-Prinzip).

### 0.1 Basis-Ordnerstruktur auf dem Server anlegen (Zuerst ausführen!)

Logge dich auf dem Server ein oder führe den folgenden Befehl remote aus, um alle notwendigen Zielverzeichnisse für Logs, Modelle, Plots und Daten anzulegen:

```bash
ssh fisc4884@hpc3.hs-flensburg.de "mkdir -p ~/master-thesis/scripts ~/master-thesis/data/{corpus/{1_aligned_urls,2_raw_scraped,3_content_extracted,4_normalized_clean},lebenshilfe/texts_lebenshilfe,analysis/textcomplexityde,evaluation_sets,vocabs,synthetic,dpo,metric_weights_exp} ~/master-thesis/results/{models/{sft,dpo,decoder_only/{sft,dpo,ppo},ppo/seq2seq,loss_aggregation_exp/{dpo_sum,dpo_mean},experiments/synthetic_regressor},logs/{run_pipeline,experiments/{benchmark,context_length_ablation,data_scaling,decoder_only,dpo_beta_sweep,factuality_metric,glossary,length_bias,loss_aggregation,metric_weights,ppo,rnn_baseline,rule_adherence,sft_scaling,synthetic_regressor,textcomplexityde,token_length}},plots/{run_pipeline,experiments/{benchmark,context_length_ablation,decoder_only,dpo_beta_sweep,factuality_metric,glossary,length_bias,loss_aggregation,metric_weights,ppo,rnn_baseline,rule_adherence,sft_scaling,synthetic_regressor,textcomplexityde,token_length}},evaluation}"
```

---

### 0.2 Synchronisation nur der benötigten Quelldateien (Whitelist)

Übertrage lokal ausschließlich die Skripte, Abhängigkeiten und die unveränderlichen Rohdaten/Benchmarks auf den Server:

#### Verbund-Sync (Alle benötigten Quelldaten auf einmal übertragen):

```bash
# 1. Skripte & Abhängigkeiten
rsync -avz ./scripts/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/scripts/

rsync -avz ./requirements.txt fisc4884@hpc3.hs-flensburg.de:~/master-thesis/requirements.txt

# 2. Lokale Lebenshilfe-Rohdateien & externe Benchmarks
rsync -avz ./data/lebenshilfe/texts_lebenshilfe/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/data/lebenshilfe/texts_lebenshilfe/

rsync -avz ./data/analysis/textcomplexityde/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/data/analysis/textcomplexityde/

rsync -avz ./data/evaluation_sets/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/data/evaluation_sets/
```

#### Optional: Bereits vorhandenes URL-Alignment / Raw Scraping mitnehmen

Falls das zeitaufwändige Web-Crawling (Schritt 01/02) nicht erneut ausgeführt werden soll:

```bash
rsync -avz ./data/corpus/1_aligned_urls/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/data/corpus/1_aligned_urls/

rsync -avz ./data/corpus/2_raw_scraped/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/data/corpus/2_raw_scraped/
```

---

## 1. Setup & Installation (Python Environment)

Auf dem HPC-Server ausführen:

```bash
cd ~/master-thesis

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

## 2. Start-Befehle (All-in-One & Pipeline-Ausführung)

### 2.1 All-in-One: Gesamt-Orchestrierung (Pipeline + Alle 17 Experimente)

Startet die gesamte Hauptpipeline sowie **sämtliche 17 wissenschaftlichen Experimente und Benchmarks** vollautomatisch mit einer optimierten Slurm-Job-Abhängigkeitskette (`--dependency=afterok`):

```bash
# Standard: Startet ab Schritt 02 (da URL-Alignment bereits auf dem Server liegt)
bash scripts/sbatch/run_all_pipeline_and_experiments.sh

# Vollständig: Startet ab Schritt 01 (inkl. komplettem Web-Crawling & URL-Alignment)
bash scripts/sbatch/run_all_pipeline_and_experiments.sh --from-step-1
```

---

### 2.2 Hauptpipeline (13 Schritte End-to-End)

Startet ausschließlich die 13 Schritte der linearen Produktions-Pipeline (ohne die zusätzlichen wissenschaftlichen Ablationsstudien):

```bash
# Vollständige Pipeline ab Schritt 01:
bash scripts/sbatch/run_pipeline/run_all_pipeline.sh

# Pipeline ab Schritt 02 (wenn URL-Alignment bereits vorliegt):
bash scripts/sbatch/run_pipeline/run_all_pipeline_from_02.sh
```

---

### 2.3 Modulare Ausführung nach Themenbereichen

Falls nur bestimmte Phasen der Pipeline gerechnet oder getestet werden sollen:

| Themenbereich                   | Runner-Skript                                                 | Enthaltene Schritte   | Beschreibung                                               |
| :------------------------------ | :------------------------------------------------------------ | :-------------------- | :--------------------------------------------------------- |
| **1. Scraping & Crawling**      | `bash scripts/sbatch/run_pipeline/run_01_scraping.sh`         | 01 $\rightarrow$ 02   | URL-Alignment & Content-Extraktion (12 Quellen)            |
| **2. Lebenshilfe Vorbereitung** | `bash scripts/sbatch/run_pipeline/run_02_lebenshilfe_prep.sh` | 03 $\rightarrow$ 04   | Lokale Lebenshilfe-Texte einlesen & bereinigen             |
| **3. Korpus-Erstellung**        | `bash scripts/sbatch/run_pipeline/run_03_corpus_building.sh`  | 05, 06                | Master CSV/JSON bauen & ungesehenes 10kGNAD vorbereiten    |
| **4. Reward- & Metrik-Modelle** | `bash scripts/sbatch/run_pipeline/run_04_reward_models.sh`    | 07, 08, 09 (parallel) | Satz-/Artikel-Klassifikatoren & MixUp-Regressor trainieren |
| **5. SFT-Training**             | `bash scripts/sbatch/run_pipeline/run_05_sft_training.sh`     | 10                    | Supervised Fine-Tuning von mBART-50                        |
| **6. DPO-Pipeline**             | `bash scripts/sbatch/run_pipeline/run_06_dpo_pipeline.sh`     | 11 $\rightarrow$ 12   | Self-Play Preference-Data Generierung & LoRA DPO-Training  |
| **7. Pipeline-Evaluierung**     | `bash scripts/sbatch/run_pipeline/run_07_evaluation.sh`       | 13                    | Finale Benchmark-Evaluierung auf Lebenshilfe               |

---

## 3. Übersicht der 13 Pipeline-Einzelschritte

| Schritt | Skriptname                         | Ressource                | Max. Zeit | Beschreibung & Datensatz                                    |
| :------ | :--------------------------------- | :----------------------- | :-------- | :---------------------------------------------------------- |
| **01**  | `01_crawl_url_alignment.sh`        | CPU (4 Cores, 16GB)      | 4h        | Stufe 1: URL-Alignment aller 12 Webquellen                  |
| **02**  | `02_crawl_content_extraction.sh`   | CPU (4 Cores, 16GB)      | 4h        | Stufe 2: HTML-Text- und Content-Extraktion                  |
| **03**  | `03_create_lebenshilfe_dataset.sh` | CPU (4 Cores, 16GB)      | 4h        | Lebenshilfe-Rohdokumente einlesen                           |
| **04**  | `04_clean_lebenshilfe.sh`          | CPU (4 Cores, 16GB)      | 4h        | Lebenshilfe Bereinigung (Signaturen & Footer entfernen)     |
| **05**  | `05_build_corpus_master.sh`        | GPU (`gpu:1`, dynamisch) | 4h        | Filterung, Deduplizierung & Master CSV/JSON Erstellung      |
| **06**  | `06_prepare_10kgnad_dpo_corpus.sh` | CPU (4 Cores, 16GB)      | 1h        | Bereinigung des 10kGNAD Alltagssprache-Korpus für DPO       |
| **07**  | `07_train_sentence_classifier.sh`  | GPU (`gpu:1`, dynamisch) | 4h        | BiLSTM Satz-Klassifikator trainieren                        |
| **08**  | `08_train_article_classifier.sh`   | GPU (`gpu:1`, dynamisch) | 4h        | BiLSTM Artikel-Klassifikator trainieren                     |
| **09**  | `09_train_mixup_regressor.sh`      | GPU (`gpu:1`, dynamisch) | 4h        | BiLSTM MixUp-Regressor Style-Score Modell trainieren        |
| **10**  | `10_train_sft.sh`                  | GPU (`gpu:1`, dynamisch) | 12h       | mBART-50 SFT Training auf `corpus_master.json`              |
| **11**  | `11_generate_dpo_dataset.sh`       | GPU (`gpu:1`, dynamisch) | 12h       | DPO-Paare offline erzeugen auf 10kGNAD (Temperature Ladder) |
| **12**  | `12_train_dpo.sh`                  | GPU (`gpu:1`, dynamisch) | 12h       | Seq2Seq LoRA DPO Training (mBART-50)                        |
| **13**  | `13_evaluate_pipeline.sh`          | GPU (`gpu:1`, dynamisch) | 2h        | Finale Benchmark-Evaluierung auf Lebenshilfe                |

---

## 4. Experimente & Ablationen (`scripts/sbatch/experiments/`)

Alle 17 wissenschaftlichen Experimente können entweder über den Gesamt-Orchestrierer (`run_all_pipeline_and_experiments.sh`) oder einzeln über ihre jeweiligen Runner ausgeführt werden:

| Nr.    | Experiment-Track                          | Verzeichnis / Runner                                                                         | Beschreibung                                                                                             |
| :----- | :---------------------------------------- | :------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------- |
| **01** | **Factuality & Halluzinations-Benchmark** | `sbatch scripts/sbatch/experiments/factuality_metric/1_run_factuality_metric_experiment.sh`  | Vergleich von SBERT vs. NLI vs. NER vs. Number Regex mit ROC-AUC                                         |
| **02** | **Hurraki Glossar-Extraktion**            | `sbatch scripts/sbatch/experiments/glossary/1_build_glossary.sh`                             | Extraktion komplexer Begriffe und automatische Definitionssuche via Hurraki-API                          |
| **03** | **Jina Kontextlängen-Ablation**           | `sbatch scripts/sbatch/experiments/context_length_ablation/1_run_context_length_ablation.sh` | Analyse von Truncation-Artefakten (128 vs 256 vs 512 vs 1024 vs 8192)                                    |
| **04** | **Length-Bias & Shortcut Analyse**        | `sbatch scripts/sbatch/experiments/length_bias/1_check_length_bias.sh`                       | Prüfung des Klassifikators/Regressors auf Längenkorrelationen und Artefakte                              |
| **05** | **TextComplexityDE Validierung**          | `sbatch scripts/sbatch/experiments/textcomplexityde/1_evaluate_textcomplexityde.sh`          | Externe Validierung des MixUp-Regressors auf dem standardisierten Benchmark                              |
| **06** | **RNN Baseline vs. BiLSTM**               | `sbatch scripts/sbatch/experiments/rnn_baseline/1_train_rnn_baseline.sh`                     | Vergleich des BiLSTM-Regressors mit Vanilla Elman-RNN und Unidirektionalem LSTM                          |
| **07** | **SFT Data Scaling Curve**                | `bash scripts/sbatch/experiments/sft_scaling/run_all_sft_scaling.sh`                         | Skalierungskurven des SFT-Modells über Datensatzgrößen (250 bis 2000 Artikel)                            |
| **08** | **MixUp Data Scaling Grid**               | `bash scripts/sbatch/experiments/data_scaling/run_all_data_scaling.sh`                       | 2D-Grid-Skalierung über Mischungsanzahlen (1k–20k) und Artikelanzahlen                                   |
| **09** | **Synthetischer Regressor**               | `bash scripts/sbatch/experiments/synthetic_regressor/run_all_synthetic_pipeline.sh`          | 7-stufige Pipeline mit LLM-generierten Zwischenstufen und kontinuierlichem Regressor                     |
| **10** | **Token Length Ablation (256/512/1024)**  | `bash scripts/sbatch/experiments/token_length/run_all_token_experiments.sh`                  | Untersuchung der Sequenzlänge auf Reward-Modell, SFT-Übersetzung und DPO                                 |
| **11** | **Quantitative Regel-Adhärenz**           | `sbatch scripts/sbatch/experiments/rule_adherence/1_measure_rule_adherence.sh`               | Messung formaler Leichte-Sprache-Regeln (Satzlänge, Silben, Passiv, Genitive)                            |
| **12** | **Decoder-Only Pipeline (Qwen 2.5)**      | `bash scripts/sbatch/experiments/decoder_only/run_all_decoder_only.sh`                       | Vollständige Qwen2.5-1.5B Kette: SFT $\rightarrow$ DPO-Gen $\rightarrow$ DPO-Training $\rightarrow$ Eval |
| **13** | **Metric Weighting Grid**                 | `bash scripts/sbatch/experiments/metric_weights/run_all_metric_weights_experiments.sh`       | Grid-Search über Belohnungsgewichte ($0.5/0.5$, $0.7/0.3$, $1.0/0.0$) in DPO                             |
| **14** | **Loss Aggregation (Sum vs. Mean)**       | `bash scripts/sbatch/experiments/loss_aggregation/run_all_loss_aggregation_experiments.sh`   | Vergleich von aufsummierten vs. gemittelten Log-Probabilities beim DPO-Loss                              |
| **15** | **DPO Beta Parameter Sweep**              | `bash scripts/sbatch/experiments/dpo_beta_sweep/run_all_beta_sweep.sh`                       | Parameter-Sweep über $\beta \in \{0.01, 0.05, 0.10, 0.20, 0.50\}$ mit mBART-50                           |
| **16** | **PPO Reinforcement Learning**            | `bash scripts/sbatch/experiments/ppo/run_all_ppo_experiments.sh`                             | Online PPO-Training für Seq2Seq (mBART-50) und Decoder-Only (Qwen 2.5)                                   |
| **17** | **Grand Master 5-Wege-Benchmark**         | `sbatch scripts/sbatch/experiments/benchmark/1_run_all_models_benchmark.sh`                  | Gesamtvergleich aller Modellfamilien (mBART SFT/DPO vs. Qwen SFT/DPO vs. Baseline)                       |

---

## 5. Visualisierungen & Diagramme (`scripts/visualization/`)

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

---

## 6. Cluster-Architektur & Dynamische Lastverteilung

Das Slurm-Cluster besteht aus 2 GPU-Knoten mit insgesamt **7 GPU-Slots (286 GB VRAM)**:

| Node               | GPUs / Partitionen          | VRAM pro Slot               | Zugewiesene Job-Klassen                                                   |
| :----------------- | :-------------------------- | :-------------------------- | :------------------------------------------------------------------------ |
| **`hpc3-perlman`** | 4x MIG Slices               | **24 GB** pro Slot          | Standard-GPU-Jobs (`--gres=gpu:1`)                                        |
| **`hpc3-liskov`**  | 2x MIG Slices + 1x Voll-GPU | **2x 47 GB** + **1x 96 GB** | Standard-GPU-Jobs (`--gres=gpu:1`) & Heavy Jobs (`--gres=gpu:mig_48gb:1`) |

_Durch die Verwendung von `#SBATCH --gres=gpu:1` verteilt Slurm alle Standard-Jobs automatisch über sämtliche 7 Slots beider Server, wodurch bis zu 7 GPU-Jobs parallel rechnen können._

---

## 7. Slurm Cluster Monitoring & Troubleshooting

```bash
# Aktuell laufende und wartende Jobs mit Node-Zuweisung anzeigen:
squeue -u $USER -o "%.10i %.9P %.30j %.8u %.2t %.10M %.6D %R %N"

# Detaillierter Status inklusive Job-Abhängigkeiten:
squeue -u $USER -o "%.10i %.9P %.30j %.8u %.2t %.10M %.6D %R %E"

# Live-Logs überwachen:
tail -f results/logs/run_pipeline/*.out
tail -f results/logs/experiments/*/*.out

# Alle eigenen Jobs abbrechen:
scancel -u $USER
```
