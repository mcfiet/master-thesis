# Pipeline SBATCH Skripte

Übersicht aller Slurm `sbatch` Ausführungsskripte für die gesamte Datenbeschaffungs-, Vorverarbeitungs- und Trainings-Pipeline.

Alle Skripte befinden sich in diesem Ordner (`scripts/sbatch/run_pipeline/`) und können aus dem Wurzelverzeichnis des Repositories per `sbatch scripts/sbatch/run_pipeline/<skript_name>.sh` gestartet werden.

---

## Übersicht der Pipeline-Schritte

| Nr. | Skriptname | SBATCH Job-Name | Ressource | Max. Zeit | Beschreibung |
|:---|:---|:---|:---|:---|:---|
| **01** | [`1_crawl_url_alignment.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/1_crawl_url_alignment.sh) | `1_crawl_url_alignment` | CPU (4 Cores, 16GB) | 4h | Stufe 1: URL-Alignment aller 12 Webquellen |
| **02** | [`2_crawl_content_extraction.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/2_crawl_content_extraction.sh) | `2_crawl_content_extraction` | CPU (4 Cores, 16GB) | 4h | Stufe 2: Text- und Content-Extraktion |
| **03** | [`3_create_lebenshilfe_dataset.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/3_create_lebenshilfe_dataset.sh) | `3_create_lebenshilfe_dataset` | CPU (4 Cores, 16GB) | 4h | Lebenshilfe-Rohdaten einlesen |
| **04** | [`4_clean_lebenshilfe.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/4_clean_lebenshilfe.sh) | `4_clean_lebenshilfe` | CPU (4 Cores, 16GB) | 4h | Lebenshilfe Bereinigung (Signaturen etc.) |
| **05** | [`5_build_corpus_master.sh`](5_build_corpus_master.sh) | `5_build_corpus_master` | GPU (MIG 24GB) | 4h | **Konsolidiert:** Filterung, Deduplizierung & Master CSV/JSON (SBERT, Spacy) |
| **06** | [`6_generate_synthetic_steps_lh.sh`](6_generate_synthetic_steps_lh.sh) | `6_generate_synthetic_steps_lh` | CPU (4 Cores, 16GB) | 6h | Synthetische Stufen für Lebenshilfe via LLM |
| **07** | [`7_generate_synthetic_steps_corpus.sh`](7_generate_synthetic_steps_corpus.sh) | `7_generate_synthetic_steps_corpus` | CPU (4 Cores, 16GB) | 12h | Synthetische Stufen für Web-Korpus via LLM |
| **08** | [`8_train_sentence_classifier.sh`](8_train_sentence_classifier.sh) | `8_train_sentence_classifier` | GPU (MIG 24GB) | 4h | BiLSTM Satz-Klassifikator trainieren |
| **09** | [`9_train_article_classifier.sh`](9_train_article_classifier.sh) | `9_train_article_classifier` | GPU (MIG 24GB) | 4h | BiLSTM Artikel-Klassifikator trainieren |
| **10a** | [`10a_train_mixup_regressor.sh`](10a_train_mixup_regressor.sh) | `10a_train_mixup_regressor` | GPU (MIG 24GB) | 4h | BiLSTM MixUp-Regressor Reward-Modell trainieren |
| **10b** | [`10b_train_synthetic_regressor.sh`](10b_train_synthetic_regressor.sh) | `10b_train_synthetic_regressor` | GPU (MIG 24GB) | 4h | BiLSTM Synthetischer Regressor Reward-Modell trainieren |
| **10c** | [`10c_train_rnn_baseline_regressor.sh`](10c_train_rnn_baseline_regressor.sh) | `10c_train_rnn_baseline_regressor` | GPU (MIG 24GB) | 4h | BiLSTM RNN Baseline Regressor trainieren |
| **11a** | [`11a_train_sft_mixup.sh`](11a_train_sft_mixup.sh) | `11a_train_sft_mixup` | GPU (MIG 24GB) | 12h | mBART-50 SFT Training (Eval mit MixUp Reward-Modell) |
| **11b** | [`11b_train_sft_synthetic.sh`](11b_train_sft_synthetic.sh) | `11b_train_sft_synthetic` | GPU (MIG 24GB) | 12h | mBART-50 SFT Training (Eval mit Synthetic Reward-Modell) |
| **12a** | [`12a_generate_dpo_dataset_mixup.sh`](12a_generate_dpo_dataset_mixup.sh) | `12a_generate_dpo_dataset_mixup` | GPU (MIG 24GB) | 12h | DPO-Paare offline erzeugen mit MixUp Reward-Modell |
| **12b** | [`12b_generate_dpo_dataset_synthetic.sh`](12b_generate_dpo_dataset_synthetic.sh) | `12b_generate_dpo_dataset_synthetic` | GPU (MIG 24GB) | 12h | DPO-Paare offline erzeugen mit Synthetic Reward-Modell |
| **13a** | [`13a_train_dpo_mixup.sh`](13a_train_dpo_mixup.sh) | `13a_train_dpo_mixup` | GPU (MIG 24GB) | 12h | Seq2Seq LoRA DPO Training (MixUp Daten) |
| **13b** | [`13b_train_dpo_synthetic.sh`](13b_train_dpo_synthetic.sh) | `13b_train_dpo_synthetic` | GPU (MIG 24GB) | 12h | Seq2Seq LoRA DPO Training (Synthetic Daten) |

---

## Experimente (außerhalb der Hauptpipeline)

* **Glossar-Anreicherung (Hurraki):**  
  Befindet sich in [`scripts/sbatch/experiments/glossary/`](../experiments/glossary/) und [`scripts/experiments/glossary/`](../../experiments/glossary/).

---

## Ausführung für den Web-Korpus

```bash
# 1. Content Extraction mit den neuen Scraper-Regeln:
sbatch scripts/sbatch/run_pipeline/2_crawl_content_extraction.sh

# 2. Master Corpus bauen (Filterung, Deduplizierung, Metriken):
sbatch scripts/sbatch/run_pipeline/5_build_corpus_master.sh
```
