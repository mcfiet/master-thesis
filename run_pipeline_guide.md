# Anleitung zur Ausführung der gesamten Pipeline

Diese Anleitung beschreibt alle Schritte, um die Daten-, Trainings- und Evaluierungs-Pipeline der Masterarbeit von Grund auf neu aufzusetzen, die Daten zu crawlen, vorzuverarbeiten und die Modelle (Klassifikation, MixUp-Regression, SFT, DPO) auf einem GPU-Server zu trainieren und zu evaluieren.

---

## 0. Vorbereitung & Synchronisation auf den HPC-Server (rsync)

Um die Pipeline auf dem Server auszuführen, müssen **nur die Basisdateien und Rohdaten** übertragen werden. Nicht benötigt werden:
- Lokale Umgebungen & Bytecode (`.venv`, `__pycache__`, `*.pyc`, `download`)
- Metadaten & Notizen (`.git`, `.obsidian`, `.DS_Store`, `research`)
- Lokale Web-App & LaTeX-Arbeit (`web`, `templates`, `thesis`)
- Bereits existierende Ergebnisse & Logs (`results`, `*.log`, `notebooks`)
- Generierte Zwischendaten (`data/analysis`, `data/lebenshilfe`, `data/vocabs`, `data/corpus/3_*`, `4_*`, `5_*`), da diese von den SBATCH-Skripten Schritt für Schritt selbst erzeugt werden.

---

### 0.1 Modularer Sync (Schritt für Schritt)

#### Szenario A (Komplette Pipeline inkl. neuem Web-Scraping):
```bash
# 1. Pipeline-Skripte (inkl. scripts/sbatch/run_pipeline) & Konfiguration
rsync -avz ./scripts/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/scripts/
rsync -avz ./requirements.txt fisc4884@hpc3.hs-flensburg.de:~/master-thesis/requirements.txt

# 2. Lokale Rohdaten (Lebenshilfe-Dokumente)
rsync -avz ./data/texts_lebenshilfe/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/data/texts_lebenshilfe/
```

#### Szenario B (Scraping überspringen & vorhandene Web-Rohtexte mitnehmen):
```bash
# 1. Pipeline-Skripte & Konfiguration
rsync -avz ./scripts/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/scripts/
rsync -avz ./requirements.txt fisc4884@hpc3.hs-flensburg.de:~/master-thesis/requirements.txt

# 2. Lokale Lebenshilfe-Rohdokumente
rsync -avz ./data/texts_lebenshilfe/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/data/texts_lebenshilfe/

# 3. Bereits vorhandene Web-Rohtexte (Schritt 1 & 2 überspringen)
rsync -avz ./data/corpus/2_raw_scraped/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/data/corpus/2_raw_scraped/
rsync -avz ./data/corpus/1_aligned_urls/ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/data/corpus/1_aligned_urls/
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
  --exclude='data/analysis' \
  --exclude='data/lebenshilfe' \
  --exclude='data/vocabs' \
  ./ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/
```

---

### 0.3 Basis-Ordnerstruktur auf dem Server anlegen

Logge dich auf dem Server ein und erstelle die Zielverzeichnisse für die Pipeline-Generierung:

```bash
ssh fisc4884@hpc3.hs-flensburg.de "mkdir -p ~/master-thesis/data/{corpus,lebenshilfe,analysis,vocabs} ~/master-thesis/results/{models,logs,plots,evaluation}"
```

---

## 1. Setup & Installation (Python Environment)

```bash
# 1. Virtuelle Umgebung erstellen und aktivieren
python3 -m venv .venv
source .venv/bin/activate

# 2. Abhängigkeiten installieren
pip install --upgrade pip
pip install -r requirements.txt

# 3. SpaCy-Sprachmodelle herunterladen
python3 -m spacy download de_core_news_lg
python3 -m spacy download de_core_news_sm
```

---

## 2. Kanonische Standard-Pipeline (`scripts/sbatch/run_pipeline/`)

Die Pipeline besteht aus 13 sequentiell aufeinander aufbauenden Schritten:

```
[01/02 Web Scraping] & [03/04 Lebenshilfe] 
                    │
                    ▼
          [05 Corpus Master] 
                    │
   ┌────────────────┼──────────────────────────────┐
   ▼                ▼                              ▼
[06 10kGNAD DPO] [07/08 Klassifikatoren]  [09 MixUp-Regressor]
   │                                               │
   │                                               ▼
   │                                       [10 SFT Training]
   │                                               │
   └───────────────────────┬───────────────────────┘
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
| **01** | `01_crawl_url_alignment.sh` | CPU (4 Cores, 16GB) | URL-Alignment aller Webquellen |
| **02** | `02_crawl_content_extraction.sh` | CPU (4 Cores, 16GB) | HTML-Text-Extraktion |
| **03** | `03_create_lebenshilfe_dataset.sh` | CPU (4 Cores, 16GB) | Lokale Lebenshilfe-Rohdateien einlesen |
| **04** | `04_clean_lebenshilfe.sh` | CPU (4 Cores, 16GB) | Bereinigung des Lebenshilfe-Datensatzes |
| **05** | `05_build_corpus_master.sh` | GPU (MIG 24GB) | Filterung, Deduplizierung, CSV/JSON Master-Korpus |
| **06** | `06_prepare_10kgnad_dpo_corpus.sh` | CPU (4 Cores, 16GB) | 10kGNAD Alltagssprache für DPO aufbereiten |
| **07** | `07_train_sentence_classifier.sh` | GPU (MIG 24GB) | Satz-Klassifikator trainieren |
| **08** | `08_train_article_classifier.sh` | GPU (MIG 24GB) | Artikel-Klassifikator trainieren |
| **09** | `09_train_mixup_regressor.sh` | GPU (MIG 24GB) | MixUp Style-Score Regressor trainieren |
| **10** | `10_train_sft.sh` | GPU (MIG 24GB) | mBART-50 SFT Training auf `corpus_master.json` |
| **11** | `11_generate_dpo_dataset.sh` | GPU (MIG 24GB) | DPO-Paare auf ungesehenem 10kGNAD generieren |
| **12** | `12_train_dpo.sh` | GPU (MIG 24GB) | LoRA DPO Training |
| **13** | `13_evaluate_pipeline.sh` | GPU (MIG 24GB) | End-to-End Evaluierung auf dem Lebenshilfe-Benchmark |

### 2.2 Ausführung aller Schritte als Slurm-Job-Kette

Um die gesamte Pipeline automatisiert mit Job-Abhängigkeiten (`--dependency=afterok`) zu starten:

```bash
bash scripts/sbatch/run_pipeline/run_all_pipeline.sh
```

---

## 3. Experimente & Ablationen (`scripts/sbatch/experiments/`)

Experimente sind modular vom Haupt-Pipeline-Ablauf getrennt:

1. **Synthetischer Regressor (`scripts/sbatch/experiments/synthetic_regressor/`):**
   - Generierung kontinuierlicher Zwischenstufen via LLM API (`FlensGen-GPT-OSS-120B`) und Training eines synthetischen BiLSTM Reward-Modells.
   - Master-Runner: `bash scripts/sbatch/experiments/synthetic_regressor/run_all_synthetic_pipeline.sh`
2. **RNN Baseline (`scripts/sbatch/experiments/rnn_baseline/`):**
   - Vergleich von BiLSTM mit Vanilla Elman-RNN und Unidirektionalem LSTM.
3. **Loss Aggregation (`scripts/sbatch/experiments/loss_aggregation/`):**
   - Untersuchung von `mean` vs. `sum` DPO-Loss.
4. **Metric Weights (`scripts/sbatch/experiments/metric_weights/`):**
   - Grid-Search über Gewichte von Style- vs. Semantik-Reward ($w_{\text{style}}, w_{\text{sem}}$).
5. **Token Length (`scripts/sbatch/experiments/token_length/`):**
   - Skalierungsexperimente bei Sequenzlängen von 256, 500 und 1000 Tokens.
