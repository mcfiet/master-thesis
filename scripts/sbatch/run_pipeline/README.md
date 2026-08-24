# Pipeline SBATCH Skripte (`scripts/sbatch/run_pipeline/`)

Übersicht aller Slurm `sbatch` Ausführungsskripte für die standardmäßige, lineare End-to-End Datenbeschaffungs-, Vorverarbeitungs-, Trainings- und Evaluierungs-Pipeline.

Alle Skripte befinden sich in diesem Ordner und können aus dem Wurzelverzeichnis des Repositories per `sbatch scripts/sbatch/run_pipeline/<skript_name>.sh` oder vollständig automatisiert über `scripts/sbatch/run_pipeline/run_all_pipeline.sh` gestartet werden.

---

## Übersicht der 13 Pipeline-Schritte

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

## Vollständige Ausführung mit einem Befehl

```bash
bash scripts/sbatch/run_pipeline/run_all_pipeline.sh
```

---

## Experimente & Ablationen (`scripts/sbatch/experiments/`)

* **Synthetischer Regressor:** [`scripts/sbatch/experiments/synthetic_regressor/`](../experiments/synthetic_regressor/)
* **RNN Baseline Regressor:** [`scripts/sbatch/experiments/rnn_baseline/`](../experiments/rnn_baseline/)
* **Loss Aggregation:** [`scripts/sbatch/experiments/loss_aggregation/`](../experiments/loss_aggregation/)
* **Metric Weights:** [`scripts/sbatch/experiments/metric_weights/`](../experiments/metric_weights/)
* **Data & SFT Scaling:** [`scripts/sbatch/experiments/sft_scaling/`](../experiments/sft_scaling/) und [`scripts/sbatch/experiments/data_scaling/`](../experiments/data_scaling/)
* **Token Length (256 vs 500 vs 1000):** [`scripts/sbatch/experiments/token_length/`](../experiments/token_length/)
