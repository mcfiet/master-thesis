# Pipeline SBATCH Skripte (`scripts/sbatch/run_pipeline/`)

Übersicht aller Slurm `sbatch` Ausführungsskripte für die standardmäßige, lineare End-to-End Datenbeschaffungs-, Vorverarbeitungs-, Trainings- und Evaluierungs-Pipeline.

Alle Skripte befinden sich in diesem Ordner und können aus dem Wurzelverzeichnis des Repositories per `sbatch scripts/sbatch/run_pipeline/<skript_name>.sh`, themenbereichsweise modular über die `run_0*_*.sh` Runner oder vollständig automatisiert über `run_all_pipeline.sh` gestartet werden.

---

## Übersicht der 13 Pipeline-Einzelschritte

| Nr. | Skriptname | SBATCH Job-Name | Ressource | Max. Zeit | Beschreibung & Datensatz |
|:---|:---|:---|:---|:---|:---|
| **01** | [`01_crawl_url_alignment.sh`](01_crawl_url_alignment.sh) | `01_crawl_url_alignment` | CPU (4 Cores, 16GB) | 4h | Stufe 1: URL-Alignment aller Webquellen |
| **02** | [`02_crawl_content_extraction.sh`](02_crawl_content_extraction.sh) | `02_crawl_content_extraction` | CPU (4 Cores, 16GB) | 4h | Stufe 2: Text- und Content-Extraktion |
| **03** | [`03_create_lebenshilfe_dataset.sh`](03_create_lebenshilfe_dataset.sh) | `03_create_lebenshilfe_dataset` | CPU (4 Cores, 16GB) | 4h | Lebenshilfe-Rohdokumente einlesen |
| **04** | [`04_clean_lebenshilfe.sh`](04_clean_lebenshilfe.sh) | `04_clean_lebenshilfe` | CPU (4 Cores, 16GB) | 4h | Lebenshilfe Bereinigung (Signaturen & Footer entfernen) |
| **05** | [`05_build_corpus_master.sh`](05_build_corpus_master.sh) | `05_build_corpus_master` | GPU (MIG 24GB) | 4h | Filterung, Deduplizierung & Master CSV/JSON Erstellung |
| **06** | [`06_prepare_10kgnad_dpo_corpus.sh`](06_prepare_10kgnad_dpo_corpus.sh) | `06_prepare_10kgnad_dpo_corpus` | CPU (4 Cores, 16GB) | 1h | Bereinigung des 10kGNAD Alltagssprache-Korpus für DPO |
| **07** | [`07_train_sentence_classifier.sh`](07_train_sentence_classifier.sh) | `07_train_sentence_classifier` | GPU (MIG 24GB) | 4h | BiLSTM Satz-Klassifikator trainieren |
| **08** | [`08_train_article_classifier.sh`](08_train_article_classifier.sh) | `08_train_article_classifier` | GPU (MIG 24GB) | 4h | BiLSTM Artikel-Klassifikator trainieren |
| **09** | [`09_train_mixup_regressor.sh`](09_train_mixup_regressor.sh) | `09_train_mixup_regressor` | GPU (MIG 24GB) | 4h | BiLSTM MixUp-Regressor Style-Score Modell trainieren |
| **10** | [`10_train_sft.sh`](10_train_sft.sh) | `10_train_sft` | GPU (MIG 24GB) | 12h | mBART-50 SFT Training auf `corpus_master.json` |
| **11** | [`11_generate_dpo_dataset.sh`](11_generate_dpo_dataset.sh) | `11_generate_dpo_dataset` | GPU (MIG 24GB) | 12h | DPO-Paare offline erzeugen auf 10kGNAD (Temperature Ladder + Anti-Repetition) |
| **12** | [`12_train_dpo.sh`](12_train_dpo.sh) | `12_train_dpo` | GPU (MIG 24GB) | 12h | Seq2Seq LoRA DPO Training |
| **13** | [`13_evaluate_pipeline.sh`](13_evaluate_pipeline.sh) | `13_evaluate_pipeline` | GPU (MIG 24GB) | 2h | Finale Benchmark-Evaluierung auf ungesehenem Lebenshilfe-Testset |

---

## Modulare Themenbereich-Runner (Teil für Teil starten)

Neben der Gesamtausführung können einzelne Themenbereiche isoliert ausgeführt werden:

| Themenbereich | Runner-Skript | Enthaltene Schritte | Beschreibung |
|:---|:---|:---|:---|
| **1. Scraping & Crawling** | [`run_01_scraping.sh`](run_01_scraping.sh) | 01 $\rightarrow$ 02 | URL-Alignment & Content-Extraktion für alle 12 Webquellen |
| **2. Lebenshilfe Vorbereitung** | [`run_02_lebenshilfe_prep.sh`](run_02_lebenshilfe_prep.sh) | 03 $\rightarrow$ 04 | Lokale Lebenshilfe-Texte einlesen & bereinigen |
| **3. Korpus-Erstellung** | [`run_03_corpus_building.sh`](run_03_corpus_building.sh) | 05, 06 | Master CSV/JSON bauen & ungesehenes 10kGNAD vorbereiten |
| **4. Reward- & Metrik-Modelle** | [`run_04_reward_models.sh`](run_04_reward_models.sh) | 07, 08, 09 (parallel) | Satz-/Artikel-Klassifikatoren & MixUp-Regressor trainieren |
| **5. SFT-Training** | [`run_05_sft_training.sh`](run_05_sft_training.sh) | 10 | Supervised Fine-Tuning von mBART-50 |
| **6. DPO-Pipeline** | [`run_06_dpo_pipeline.sh`](run_06_dpo_pipeline.sh) | 11 $\rightarrow$ 12 | Preference-Data Generierung & LoRA DPO-Training |
| **7. Pipeline-Evaluierung** | [`run_07_evaluation.sh`](run_07_evaluation.sh) | 13 | Finale Evaluierung auf dem Lebenshilfe-Benchmark |

### Beispiele für Modulare Aufrufe

```bash
# Nur Web Scraping ausführen:
bash scripts/sbatch/run_pipeline/run_01_scraping.sh

# Nur Reward-Modelle trainieren (nachdem Korpus vorliegt):
bash scripts/sbatch/run_pipeline/run_04_reward_models.sh

# Nur DPO-Generierung und DPO-Training starten:
bash scripts/sbatch/run_pipeline/run_06_dpo_pipeline.sh
```

---

## Vollständige Ausführung mit einem Befehl

Startet alle 13 Schritte sequentiell mit automatischer Slurm-Abhängigkeitskette:

```bash
bash scripts/sbatch/run_pipeline/run_all_pipeline.sh
```

Falls das URL-Alignment (Schritt 01) bereits ausgeführt wurde und auf dem Server vorliegt, kann die Pipeline direkt ab Schritt 02 gestartet werden:

```bash
bash scripts/sbatch/run_pipeline/run_all_pipeline_from_02.sh
```

---

## Gesamt-Orchestrierung: Pipeline + Alle 17 Experimente mit einem Befehl

Startet die Hauptpipeline sowie sämtliche 17 Experimente, Ablationen und den 5-Wege-Benchmark mit vollständiger Slurm-Abhängigkeitskette:

```bash
# Standard (startet Hauptpipeline ab Schritt 02):
bash scripts/sbatch/run_all_pipeline_and_experiments.sh

# Oder ab Schritt 01 (inkl. neuem URL-Crawling):
bash scripts/sbatch/run_all_pipeline_and_experiments.sh --from-step-1
```

---

## Experimente & Ablationen (`scripts/sbatch/experiments/`)

### Metrik & Reward-Modelle (`scripts/sbatch/experiments/metric/`)
* **Klassifikator-Längen (256/512/1024):** [`scripts/sbatch/experiments/metric/classifier_length/`](../experiments/metric/classifier_length/)
* **Regressor-Längen (256/512/1024):** [`scripts/sbatch/experiments/metric/regressor_length/`](../experiments/metric/regressor_length/)
* **MixUp-Varianten (Static/Dynamic/Hybrid/Cyclic):** [`scripts/sbatch/experiments/metric/mixup_variants/`](../experiments/metric/mixup_variants/)
* **RNN Baseline Regressor:** [`scripts/sbatch/experiments/metric/rnn_baseline/`](../experiments/metric/rnn_baseline/)
* **TextComplexityDE Externe Validierung:** [`scripts/sbatch/experiments/metric/textcomplexityde/`](../experiments/metric/textcomplexityde/)
* **MixUp Data Scaling:** [`scripts/sbatch/experiments/metric/data_scaling/`](../experiments/metric/data_scaling/)
* **Synthetischer Regressor:** [`scripts/sbatch/experiments/metric/synthetic_regressor/`](../experiments/metric/synthetic_regressor/)
* **Length Bias Analyse:** [`scripts/sbatch/experiments/metric/length_bias/`](../experiments/metric/length_bias/)
* **Jina Kontextlängen-Ablation:** [`scripts/sbatch/experiments/metric/context_length_ablation/`](../experiments/metric/context_length_ablation/)
* **Faktentreue & Halluzinationserkennung:** [`scripts/sbatch/experiments/metric/factuality_metric/`](../experiments/metric/factuality_metric/)
* **Similarity Threshold Ablation:** [`scripts/sbatch/experiments/metric/similarity_threshold/`](../experiments/metric/similarity_threshold/)
* **Längenstudien Master-Runner:** [`scripts/sbatch/experiments/metric/run_all_length_experiments.sh`](../experiments/metric/run_all_length_experiments.sh)

### Übersetzung, Alignment & Benchmarks (`scripts/sbatch/experiments/`)
* **Decoder-Only Modelle (Qwen2.5):** [`scripts/sbatch/experiments/decoder_only/`](../experiments/decoder_only/)
* **DPO Beta Parameter Sweep:** [`scripts/sbatch/experiments/dpo_beta_sweep/`](../experiments/dpo_beta_sweep/)
* **Loss Aggregation (mean vs sum):** [`scripts/sbatch/experiments/loss_aggregation/`](../experiments/loss_aggregation/)
* **Metric Weights Grid:** [`scripts/sbatch/experiments/metric_weights/`](../experiments/metric_weights/)
* **PPO Reinforcement Learning:** [`scripts/sbatch/experiments/ppo/`](../experiments/ppo/)
* **SFT Data Scaling:** [`scripts/sbatch/experiments/sft_scaling/`](../experiments/sft_scaling/)
* **Token Length (256 vs 512 vs 1024):** [`scripts/sbatch/experiments/token_length/`](../experiments/token_length/)
* **Google mT5-base Pipeline:** [`scripts/sbatch/experiments/run_mt5_pipeline.sh`](../experiments/run_mt5_pipeline.sh)
* **Master 5-Wege-Benchmark:** [`scripts/sbatch/experiments/benchmark/`](../experiments/benchmark/)
* **Quantitative Regeltreue (Rule Adherence):** [`scripts/sbatch/experiments/rule_adherence/`](../experiments/rule_adherence/)
* **Glossar-Extraktion:** [`scripts/sbatch/experiments/glossary/`](../experiments/glossary/)

