# Anleitung zur Ausführung der gesamten Pipeline

Diese Anleitung beschreibt alle Schritte, um die Daten- und Modellierungs-Pipeline der Masterarbeit von Grund auf neu aufzusetzen, die Daten zu crawlen, vorzuverarbeiten und die Modelle (Klassifikation, Regression, Übersetzung) auf einem GPU-Server zu trainieren.

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

#### All-in-One: Szenario B (Scraping überspringen, Rohtexte behalten)
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
  --exclude='data/corpus/3_*' \
  --exclude='data/corpus/4_*' \
  --exclude='data/corpus/5_*' \
  --exclude='data/corpus/*with_steps*' \
  --exclude='data/analysis' \
  --exclude='data/lebenshilfe' \
  --exclude='data/vocabs' \
  ./ fisc4884@hpc3.hs-flensburg.de:~/master-thesis/
```

---

### 0.3 Basis-Ordnerstruktur auf dem Server anlegen

Logge dich auf dem Server ein und erstelle die Zielverzeichnisse für die Pipeline-Generierung:

```bash
ssh fisc4884@hpc3.hs-flensburg.de "mkdir -p ~/master-thesis/data/{corpus,lebenshilfe,analysis,vocabs} ~/master-thesis/results/{models,logs,plots}"
```

---

## 1. Setup & Installation (Python Environment)

### 1.1 Virtuelle Umgebung erstellen und aktivieren

Führe diese Befehle im Hauptverzeichnis des Repositories aus:

```bash
# Virtuelle Umgebung erstellen
python3 -m venv .venv

# Aktivieren (macOS / Linux)
source .venv/bin/activate
```

### 1.2 Python-Abhängigkeiten installieren

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.3 SpaCy-Sprachmodelle herunterladen

Das Projekt benötigt deutsche Sprachmodelle für Named Entity Recognition (NER) und Satzsegmentierung:

```bash
# Großes Modell (für genaue Analysen und NER)
python3 -m spacy download de_core_news_lg

# Kleines Modell (für schnellere Satzanalysen)
python3 -m spacy download de_core_news_sm
```

---

## 2. Scraping & Datenbeschaffung (Schritt 1)

Der Korpus wird in einer zweistufigen Pipeline aus 12 deutschen Webquellen aufgebaut:

### 2.1 Stufe 1: URL-Alignment (URL-Paare finden)

Sucht auf den Ziel-Webseiten nach Artikeln in Leichter Sprache (LS) und deren alltagssprachlicher Entsprechung (AS). Speichert die Ergebnisse unter `data/corpus/aligned_urls/<quelle>_aligned_urls.json`.

```bash
# Beispiel für eine Quelle (mdr)
.venv/bin/python scripts/data_collection/crawl_scraper/mdr_scraper.py
```

_(Wiederhole dies für alle gewünschten Scraper im Ordner `scripts/data_collection/crawl_scraper/`)_

### 2.2 Stufe 2: Content-Extraction (Fließtexte extrahieren)

Lädt den HTML-Inhalt der gepaarten URLs herunter, filtert Boilerplate (Navigation, Ads, Footer) und speichert die Texte unter `data/corpus/raw/<quelle>_articles.json`.

```bash
# Beispiel für eine Quelle (mdr)
.venv/bin/python scripts/data_collection/corpus_scrapers/mdr_scraper.py
```

_(Wiederhole dies für alle Scraper im Ordner `scripts/data_collection/corpus_scrapers/`)_

---

## 3. Vorverarbeitung & Datensatzerstellung (Schritt 2)

Führe diese Skripte in der angegebenen Reihenfolge aus:

```bash
# 1. Lokales Lebenshilfe-Dokumentenkorpus einlesen und JSON erstellen
.venv/bin/python scripts/preprocessing/create_lebenshilfe_dataset.py

# 2. Lebenshilfe-Datensatz bereinigen (Metadaten, Prüferhinweise, Signaturen entfernen)
.venv/bin/python scripts/preprocessing/clean_lebenshilfe.py \
    --input_file data/lebenshilfe/lebenshilfe_dataset.json \
    --output_file data/lebenshilfe/lebenshilfe_dataset_clean.json

# 3. Information Loss & Ähnlichkeits-Analyse berechnen (erzeugt die CSV für den Filter)
.venv/bin/python scripts/evaluation/measure_information_loss.py \
    --input_dir data/corpus/2_raw_scraped \
    --output_csv data/analysis/information_loss_analysis.csv

# 4. Web-Korpus filtern (Ähnlichkeitsbereich 0.60 bis 0.99, Mindestlänge 10 Wörter)
.venv/bin/python scripts/preprocessing/filter_similarity.py \
    --analysis_csv data/analysis/information_loss_analysis.csv \
    --source_dir data/corpus/2_raw_scraped \
    --output_dir data/corpus/3_filtered_similarity \
    --sim_min 0.60 --sim_max 0.99 --min_ls_tokens 10

# 5. Textnormalisierung & Quellenspezifische Bereinigung (z.B. Mediopunkt-Entfernung)
.venv/bin/python scripts/preprocessing/normalize_clean.py \
    --input_dir data/corpus/3_filtered_similarity \
    --output_dir data/corpus/4_normalized_clean

# 6. Glossar-Datenbank über Hurraki-API aufbauen
.venv/bin/python scripts/preprocessing/build_glossary.py

# 7. Textkorpus mit Begriffserklärungen anreichern
.venv/bin/python scripts/preprocessing/enrich_glossary.py

# 8. Master-CSV und JSON generieren (berechnet Metriken wie Lesbarkeits-Scores und MATTR)
.venv/bin/python scripts/preprocessing/build_corpus_master.py \
    --input_dir data/corpus/4_normalized_clean \
    --output_csv data/analysis/corpus_master.csv

# 9. Synthetische Komplexitäts-Stufen (0.25, 0.50, 0.75) via LLM API erzeugen
# (Ersetze <API_URL> und <TOKEN> durch deine Verbindungsdaten)
.venv/bin/python scripts/preprocessing/generate_synthetic_steps.py \
    --input data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --output data/lebenshilfe/lebenshilfe_dataset_with_steps.json \
    --url <API_URL> --token <TOKEN> --model "FlensGen-GPT-OSS-120B"

.venv/bin/python scripts/preprocessing/generate_synthetic_steps.py \
    --input data/analysis/corpus_master.json \
    --output data/corpus/corpus_master_with_steps.json \
    --url <API_URL> --token <TOKEN> --model "FlensGen-GPT-OSS-120B"
```

---

## 4. Modelltraining (Schritt 3)

Die Skripte 1 bis 6 trainieren die Klassifikatoren, Regressoren, sowie SFT und DPO Übersetzungsmodelle.
Alle Parameter müssen per Kommandozeilenargument übergeben werden (keine Standardwerte, ansonsten Abbruch mit Fehler). Ausgaben werden live an der Konsole angezeigt und parallel unter `results/logs/` in Logdateien gesichert.

### 4.1 Metrik-Modelle (CPU-freundlich / GPU optional)

Diese Modelle sind kompakt und können schnell auf Standard-CPUs trainiert werden:

```bash
# 1. Satz-Klassifikator (BiLSTM)
# [CPU Friendly - GPU Optional]
.venv/bin/python scripts/modeling/binary_train_sentence_model.py \
    --csv_path data/analysis/corpus_master.csv \
    --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --batch_size 64 --embedding_dim 128 --epochs 20 --hidden_dim 128 \
    --lr 0.001 --max_seq_len 100 --max_sim 0.98 --min_sent_len 3 --min_sim 0.8

# 2. Artikel-Klassifikator (BiLSTM)
# [CPU Friendly - GPU Optional]
.venv/bin/python scripts/modeling/binary_train_article_model.py \
    --csv_path data/analysis/corpus_master.csv \
    --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --batch_size 32 --embedding_dim 128 --epochs 30 --hidden_dim 128 \
    --lr 0.001 --max_seq_len 512 --max_sim 0.98 --min_sim 0.8

# 3. MixUp-Regressor (Style-Score)
# [CPU Friendly - GPU Optional]
.venv/bin/python scripts/modeling/regression_train_mixup.py \
    --csv_path data/analysis/corpus_master.csv \
    --batch_size 64 --embedding_dim 128 --epochs 40 --hidden_dim 128 \
    --lr 0.001 --max_sim 0.98 --min_sim 0.8 --max_seq_len 256 \
    --vocab_save_path data/vocabs/mixup_vocab.json

# 4. Synthetischer Regressor (Style-Score)
# [CPU Friendly - GPU Optional]
.venv/bin/python scripts/modeling/regression_train_synthetic.py \
    --corpus_with_steps_path data/corpus/corpus_master_with_steps.json \
    --lh_with_steps_path data/lebenshilfe/lebenshilfe_dataset_with_steps.json \
    --model_save_path results/models/bilstm_synthetic_regression.pt \
    --vocab_save_path data/vocabs/synthetic_vocab.json \
    --epochs 15 --max_seq_len 256
```

### 4.2 Übersetzungs- & DPO-Modelle (GPU zwingend erforderlich)

Das Training des mBART-50 Transformers (über 1 Mrd. Parameter) benötigt zwingend eine CUDA-fähige Grafikkarte. Die Übersetzungspipeline gliedert sich in 3 aufeinander aufbauende Schritte:

```bash
# 5. Supervised Fine-Tuning (SFT Übersetzungsmodell - 30 Epochen mit Early Stopping)
# [GPU Mandatory]
.venv/bin/python scripts/modeling/train_sft.py \
    --lh_dataset_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --corpus_path "data/analysis/corpus_master.json" \
    --output_dir "results/models/sft" \
    --min_sim 0.70 \
    --max_sim 0.98 \
    --max_source_len 256 \
    --max_target_len 256 \
    --model_name "facebook/mbart-large-50" \
    --batch_size 8 \
    --accumulation_steps 2 \
    --epochs 30 \
    --lr 1e-5 \
    --warmup_ratio 0.10 \
    --patience 5 \
    --reward_model_path "results/models/bilstm_synthetic_regression.pt" \
    --reward_vocab_path "data/vocabs/synthetic_vocab.json"

# 6. DPO-Präferenzdatensatz generieren (Offline Sampling mit Reward-Bewertung)
# [GPU Mandatory]
.venv/bin/python scripts/modeling/generate_dpo_dataset.py \
    --corpus_path "data/analysis/corpus_master.json" \
    --min_sim 0.70 \
    --max_sim 0.98 \
    --sft_model_path "results/models/sft" \
    --prompt_prefix "" \
    --num_candidates 5 \
    --temperature 0.8 \
    --reward_model_path "results/models/bilstm_synthetic_regression.pt" \
    --reward_vocab_path "data/vocabs/synthetic_vocab.json" \
    --w_style 0.5 \
    --w_sem 0.5 \
    --min_score_margin 0.05 \
    --output_file "data/dpo_preference_pairs.jsonl" \
    --val_split_ratio 0.15

# 7. Direct Preference Optimization (Natives PyTorch DPO-Training für Seq2Seq)
# [GPU Mandatory]
.venv/bin/python scripts/modeling/train_dpo.py \
    --model_name_or_path "results/models/sft" \
    --train_file "data/dpo_preference_pairs.jsonl" \
    --eval_file "data/dpo_preference_pairs_eval.jsonl" \
    --output_dir "results/models/dpo_trained_model" \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --beta 0.1 \
    --learning_rate 5e-6 \
    --epochs 3 \
    --batch_size 2 \
    --accumulation_steps 8 \
    --patience 3 \
    --max_source_len 256 \
    --max_target_len 256
```

---

## 5. Auswertung & Analyse (Research-Notebooks)

Nach erfolgreichem Training können die Modelle interaktiv in Jupyter Notebooks analysiert und visualisiert werden:

- **Daten-Analyse:**
  - [`notebooks/research/data/corpus_diagnostics.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/research/data/corpus_diagnostics.ipynb)
  - [`notebooks/research/data/analyze_boilerplate_bias.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/research/data/analyze_boilerplate_bias.ipynb)
- **Klassifikator- & Regressorbewertung:**
  - [`notebooks/research/metric/check_length_bias.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/research/metric/check_length_bias.ipynb)
  - [`notebooks/research/metric/compare_mixup_vs_synthetic.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/research/metric/compare_mixup_vs_synthetic.ipynb)
  - [`notebooks/research/metric/4_mixup_model_evaluation.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/research/metric/4_mixup_model_evaluation.ipynb)
- **Übersetzungs- & DPO-Vergleich:**
  - [`notebooks/research/translation/compare_dpo_results.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/research/translation/compare_dpo_results.ipynb)
  - [`notebooks/research/translation/compare_sft_vs_dpo_w10.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/translation/compare_sft_vs_dpo_w10.ipynb)
