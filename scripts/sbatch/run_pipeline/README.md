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
| **05** | [`5_measure_information_loss.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/5_measure_information_loss.sh) | `5_measure_information_loss` | GPU (MIG 24GB) | 4h | Information Loss & Ähnlichkeitsanalyse |
| **06** | [`6_filter_similarity.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/6_filter_similarity.sh) | `6_filter_similarity` | CPU (4 Cores, 16GB) | 4h | Filterung nach Semantischer Ähnlichkeit (0.60–0.99) |
| **07** | [`7_normalize_clean.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/7_normalize_clean.sh) | `7_normalize_clean` | CPU (4 Cores, 16GB) | 4h | Textnormalisierung & Bereinigung |
| **08** | [`8_build_glossary.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/8_build_glossary.sh) | `8_build_glossary` | CPU (4 Cores, 16GB) | 4h | Hurraki-Glossar aufbauen |
| **09** | [`9_enrich_glossary.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/9_enrich_glossary.sh) | `9_enrich_glossary` | CPU (4 Cores, 16GB) | 4h | Korpus mit Glossareinträgen anreichern |
| **10** | [`10_build_corpus_master.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/10_build_corpus_master.sh) | `10_build_corpus_master` | GPU (MIG 24GB) | 4h | Master CSV & JSON generieren (SBERT, Spacy) |
| **11** | [`11_generate_synthetic_steps_lh.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/11_generate_synthetic_steps_lh.sh) | `11_generate_synthetic_steps_lh` | CPU (4 Cores, 16GB) | 6h | Synthetische Stufen für Lebenshilfe via LLM |
| **12** | [`12_generate_synthetic_steps_corpus.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/12_generate_synthetic_steps_corpus.sh) | `12_generate_synthetic_steps_corpus` | CPU (4 Cores, 16GB) | 12h | Synthetische Stufen für Web-Korpus via LLM |
| **13** | [`13_train_sentence_classifier.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/13_train_sentence_classifier.sh) | `13_train_sentence_classifier` | GPU (MIG 24GB) | 4h | BiLSTM Satz-Klassifikator trainieren |
| **14** | [`14_train_article_classifier.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/14_train_article_classifier.sh) | `14_train_article_classifier` | GPU (MIG 24GB) | 4h | BiLSTM Artikel-Klassifikator trainieren |
| **15a** | [`15a_train_mixup_regressor.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/15a_train_mixup_regressor.sh) | `15a_train_mixup_regressor` | GPU (MIG 24GB) | 4h | BiLSTM MixUp-Regressor Reward-Modell trainieren |
| **15b** | [`15b_train_synthetic_regressor.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/15b_train_synthetic_regressor.sh) | `15b_train_synthetic_regressor` | GPU (MIG 24GB) | 4h | BiLSTM Synthetischer Regressor Reward-Modell trainieren |
| **16a** | [`16a_train_sft_mixup.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/16a_train_sft_mixup.sh) | `16a_train_sft_mixup` | GPU (Volle 96GB) | 12h | mBART-50 SFT Training (Eval mit MixUp Reward-Modell) |
| **16b** | [`16b_train_sft_synthetic.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/16b_train_sft_synthetic.sh) | `16b_train_sft_synthetic` | GPU (Volle 96GB) | 12h | mBART-50 SFT Training (Eval mit Synthetic Reward-Modell) |
| **17a** | [`17a_generate_dpo_dataset_mixup.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/17a_generate_dpo_dataset_mixup.sh) | `17a_generate_dpo_dataset_mixup` | GPU (Volle 96GB) | 12h | DPO-Paare offline erzeugen mit MixUp Reward-Modell |
| **17b** | [`17b_generate_dpo_dataset_synthetic.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/17b_generate_dpo_dataset_synthetic.sh) | `17b_generate_dpo_dataset_synthetic` | GPU (Volle 96GB) | 12h | DPO-Paare offline erzeugen mit Synthetic Reward-Modell |
| **18a** | [`18a_train_dpo_mixup.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/18a_train_dpo_mixup.sh) | `18a_train_dpo_mixup` | GPU (Volle 96GB) | 12h | Seq2Seq LoRA DPO Training (MixUp Daten) |
| **18b** | [`18b_train_dpo_synthetic.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/run_pipeline/18b_train_dpo_synthetic.sh) | `18b_train_dpo_synthetic` | GPU (Volle 96GB) | 12h | Seq2Seq LoRA DPO Training (Synthetic Daten) |

---

## Ausführung

```bash
# Beispiel: Einen einzelnen Schritt einreichen
sbatch scripts/sbatch/run_pipeline/16a_train_sft_mixup.sh

# Status der Warteschlange überprüfen
squeue -u $USER
```
