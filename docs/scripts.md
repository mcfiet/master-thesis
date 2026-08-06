# Skript-Referenz (Python-Skripte)

Diese Dokumentation beschreibt alle Python-Skripte im Ordner `scripts/`, ihre Konfigurationsparameter, Ein- und Ausgabepfade sowie Beispiele für die Ausführung.

---

## 1. Scraping-Pipeline

Die Web-Scraper arbeiten in einer zweistufigen Pipeline, um ein paralleles Korpus (Alltagssprache $\leftrightarrow$ Leichte Sprache) aus dem Web aufzubauen. Es werden insgesamt 12 verschiedene deutsche Quellen unterstützt.

### Stufe 1: Crawling & URL-Alignment (`scripts/data_collection/crawl_scraper/`)
Die Skripte in diesem Verzeichnis scannen die Ziel-Webseiten nach Artikeln in Leichter Sprache (LS) und suchen über Sprachwechsler oder Beitragslinks nach dem alltagssprachlichen Original (AS).
* **Ausgabe:** `data/corpus/aligned_urls/<quelle>_aligned_urls.json`
* **Skripte:**
  - `apotheken_scraper.py`
  - `behindertenbeauftragter_scraper.py`
  - `brandeins_scraper.py`
  - `hamburg_scraper.py`
  - `hannover_scraper.py`
  - `koeln_scraper.py`
  - `main_taunus_scraper.py`
  - `mdr_scraper.py`
  - `sozialpolitik_scraper.py`
  - `stuttgart_scraper.py`
  - `taz_scraper.py`
  - `wiesbaden_scraper.py`
* **Ausführungsbeispiel:**
  ```bash
  .venv/bin/python scripts/data_collection/crawl_scraper/mdr_scraper.py
  ```

### Stufe 2: Content-Extraktion (`scripts/data_collection/corpus_scrapers/`)
Diese Skripte lesen die aus Stufe 1 generierten URL-Paare ein, laden die Webseiten herunter, extrahieren den bereinigten Fließtext (ohne Werbung, Header, Navigation, Sidebars und Footer) und zählen die Token.
* **Eingabe:** `data/corpus/aligned_urls/<quelle>_aligned_urls.json`
* **Ausgabe:** `data/corpus/raw/<quelle>_articles.json`
* **Skripte:** Analog zur Stufe 1 benannt (z.B. `apotheken_scraper.py`).
* **Ausführungsbeispiel:**
  ```bash
  .venv/bin/python scripts/data_collection/corpus_scrapers/mdr_scraper.py
  ```

---

## 2. Lokale Datensatzerstellung

### `scripts/preprocessing/create_lebenshilfe_dataset.py`
Verarbeitet lokale Dokumentdateien (`.docx`, `.rtf`, `.odt`) der Organisation *Lebenshilfe*.
* **Funktionsweise:**
  1. Liest Dokumente aus `data/lebenshilfe/texts_lebenshilfe/as` und `data/lebenshilfe/texts_lebenshilfe/ls`.
  2. Normalisiert Dateinamen zur automatischen Paarbildung.
  3. Nutzt eine manuelle Zuordnungsliste (`manual_matches`) für Dokumente mit stark abweichenden Dateinamen.
  4. Extrahiert den Text (RTF wird zu Text konvertiert, ODT und DOCX parst das Skript strukturiert).
* **Eingabe-Verzeichnisse:** `data/lebenshilfe/texts_lebenshilfe/ls` und `data/lebenshilfe/texts_lebenshilfe/as`
* **Ausgabe-Datei:** `data/lebenshilfe/lebenshilfe_dataset.json`
* **Befehl:**
  ```bash
  .venv/bin/python scripts/preprocessing/create_lebenshilfe_dataset.py
  ```

---

## 3. Datenbereinigung & Post-Processing

### `scripts/preprocessing/clean_corpus.py`
Filtert den Korpus, um Rauschen (schlecht gemappte Artikel, Platzhalter) zu eliminieren.
* **Filterbedingungen:**
  - Semantische Ähnlichkeit (Jina 8192) zwischen $0.60$ und $0.99$.
  - Die leichtsprachliche Version muss mindestens 10 Wörter lang sein.
  - Ausschluss von "Lorem Ipsum"-Testtexten.
* **Eingabe:** `data/analysis/information_loss_analysis_cleaned.csv` & `data/corpus/raw/`
* **Ausgabe:** `data/corpus/cleaned/` (enthält gefilterte `<quelle>_articles.json`)
* **Befehl:**
  ```bash
  .venv/bin/python scripts/preprocessing/clean_corpus.py
  ```

### `scripts/preprocessing/post_clean_corpus.py`
Führt kosmetische Bereinigungen und quellenspezifische Korrekturen am gefilterten Korpus durch.
* **Reinigungs-Schritte:**
  - Entfernen des Syllable-Separators (Mediopunkt `·` oder `∙`) in LS und AS.
  - *BrandEins*: Entfernt Autorennamen und Datumszeilen am Textanfang (z.B. "März 2023.").
  - *MDR*: Entfernt Boilerplate-Footer ("Über dieses Thema berichtet der MDR auch...").
  - *Taz*: Entfernt verwaiste Bildunterschriften.
* **Eingabe:** `data/corpus/cleaned/`
* **Ausgabe:** `data/corpus/final/`
* **Befehl:**
  ```bash
  .venv/bin/python scripts/preprocessing/post_clean_corpus.py
  ```

---

## 4. Analyse von Informationsverlust & Ähnlichkeit

### `scripts/evaluation/measure_information_loss.py`
Berechnet semantische Ähnlichkeit und Named Entity Recognition (NER) Recall zwischen AS und LS.
* **Modelle:**
  - SpaCy: `de_core_news_lg` (für POS-Tagging, Satzsegmentierung und NER)
  - SBERT: `jinaai/jina-embeddings-v2-base-de` (vollständiger Kontext bis 8192 Token)
* **Berechnete Metriken:**
  - Cosine-Ähnlichkeit bei 128, 512 und 8192 Token Kontextlänge.
  - NER Recall AS $\to$ LS (Faktenerhalt) und LS $\to$ AS (Faktentreue).
  - POS-Ratios (Verhältnis von Nomen, Adjektiven, Verben, Konjunktionen).
* **Argumente:**
  - `--input_dir`: Standard: `data/corpus/raw`
  - `--output_csv`: Standard: `data/analysis/information_loss_analysis.csv`
* **Befehl (für finalen Korpus):**
  ```bash
  .venv/bin/python scripts/evaluation/measure_information_loss.py \
      --input_dir data/corpus/final \
      --output_csv data/analysis/information_loss_analysis_cleaned.csv
  ```

### `scripts/evaluation/info_loss_stats.py`
Gibt statistische Kennzahlen (Mean, Median, Standardabweichung, Min/Max) der Tokenlängen aus.
* **Eingabe:** `data/analysis/information_loss_analysis.csv`
* **Befehl:**
  ```bash
  .venv/bin/python scripts/evaluation/info_loss_stats.py
  ```

### `scripts/evaluation/calculate_sbert_coverage.py`
Prüft, wie viele Artikel über das SBERT-Tokenlimit (512) hinausgehen und abgeschnitten werden würden.
* **Eingabe:** `data/analysis/information_loss_analysis.csv`
* **Befehl:**
  ```bash
  .venv/bin/python scripts/evaluation/calculate_sbert_coverage.py
  ```

### `scripts/evaluation/count_total_tokens.py`
Zählt linguistische Tokens (mittels Regex `\w+|[^\w\s]`) über alle Roh-JSONs im Korpus.
* **Eingabe:** `data/corpus/raw/*.json`
* **Befehl:**
  ```bash
  .venv/bin/python scripts/evaluation/count_total_tokens.py
  ```

### `scripts/evaluation/summarize_corpus.py`
Gibt eine tabellarische Zusammenfassung über die Token- und Artikelpaaranzahl pro Quelle sowie deren Verhältnis (LS/AS) aus.
* **Eingabe:** `data/corpus/raw/*.json`
* **Befehl:**
  ```bash
  .venv/bin/python scripts/evaluation/summarize_corpus.py
  ```

---

## 5. Lesbarkeits- und Diversitätsmetriken

### `scripts/evaluation/corpus_stats.py`
Generiert zusammenfassende Metriken für alle Quellen (Vocab-Größe, Token/Satz-Verhältnisse etc.).
* **Argumente:**
  - `--input_dir`: Standard: `data/corpus/raw`
  - `--output_file`: Standard: `research/corpus_statistics.md`
* **Befehl:**
  ```bash
  .venv/bin/python scripts/evaluation/corpus_stats.py \
      --input_dir data/corpus/final \
      --output_file research/corpus_statistics_cleaned.md
  ```

### `scripts/evaluation/measure_readability.py`
Errechnet Lesbarkeitsindizes für Deutsch (Flesch Reading Ease via Amstad-Formel, Wiener Sachtextformel, LIX).
* **Eingabe-Pfad:** `data/corpus/final`
* **Ausgabe-Pfad:** `data/analysis/readability_analysis.csv`
* **Befehl:**
  ```bash
  .venv/bin/python scripts/evaluation/measure_readability.py
  ```

### `scripts/evaluation/measure_ttr.py`
Berechnet Type-Token-Ratio (TTR) und Moving Average Type-Token-Ratio (MATTR, Window=50) auf lemmatisierten Wörtern (ohne Satzzeichen).
* **Eingabe-Pfad:** `data/corpus/final`
* **Ausgabe-Pfad:** `data/analysis/ttr_analysis.csv`
* **Befehl:**
  ```bash
  .venv/bin/python scripts/evaluation/measure_ttr.py
  ```

---

## 6. Modellevaluierung & Klassifikation

Die Skripte prüfen die Performance trainierter PyTorch BiLSTM-Klassifikatoren (AS vs. LS).

### `scripts/modeling/evaluate_article_model.py`
Prüft das Artikel-Klassifikationsmodell auf der Lebenshilfe-Testmenge.
* **Eingabe-Datensatz:** `data/lebenshilfe/lebenshilfe_dataset_no_paragraphs.json`
* **Modell-Pfad:** `results/models/lstm_article_sim_0.80_to_0.98.pt`
* **Befehl:**
  ```bash
  .venv/bin/python scripts/modeling/evaluate_article_model.py
  ```

### `scripts/modeling/evaluate_sentence_model.py`
Prüft den Klassifikator auf Satzebene.
* **Eingabe-Datensatz:** `data/lebenshilfe/lebenshilfe_dataset_no_paragraphs.json`
* **Modell-Pfad:** `results/models/lstm_baseline_sim_0.80_to_0.98.pt`
* **Befehl:**
  ```bash
  .venv/bin/python scripts/modeling/evaluate_sentence_model.py
  ```

### `scripts/modeling/check_length_bias.py`
Überprüft statistisch und empirisch (z.B. über Dummy-Eingaben aus Punkten `.`), ob der Klassifikator die Klasse primär an der Textlänge statt an linguistischen Features festmacht.
* **Befehl:**
  ```bash
  .venv/bin/python scripts/modeling/check_length_bias.py
  ```

---

## 7. Synthetische Datengenerierung (LLM)

### `scripts/modeling/generate_synthetic_regression_steps.py`
Generiert Zwischenstufen (Standard: `0.25, 0.50, 0.75`) zwischen Leichter Sprache ($0.0$) und Alltagssprache ($1.0$) über ein OpenAI-kompatibles LLM API-Interface. Inkrementelles Schreiben verhindert Datenverluste bei Verbindungsabbrüchen.
* **Argumente:**
  * `--input`: Pfad zur Quelldatei (Standard: `data/lebenshilfe/lebenshilfe_dataset.json`)
  * `--output`: Pfad zur Zieldatei (Standard: `data/lebenshilfe/lebenshilfe_dataset_with_steps.json`)
  * `--url`: API-Endpunkt (Erforderlich)
  * `--model`: Name des LLMs (optional)
  * `--token`: Bearer Token zur Authentifizierung (optional)
  * `--steps`: Zu generierende Zielstufen (Standard: `0.25,0.50,0.75`)
  * `--limit`: Maximale Anzahl an Artikeln für Testzwecke (optional)
* **Befehl:**
  ```bash
  .venv/bin/python scripts/modeling/generate_synthetic_regression_steps.py \
      --url http://193.175.180.196:8000/v1/chat/completions \
      --limit 1 \
      --token RrI6y403jAlUm8v \
      --model "FlensGen-GPT-OSS120B"
  ```

---

## 8. Visualisierung

### `scripts/visualization/generate_review_report.py`
Erstellt einen Markdown-Report über Artikelpaare mit extremer semantischer Ähnlichkeit ($<0.6$ oder $>0.98$) zur manuellen Inspektion.
* **Eingabe:** `data/analysis/information_loss_analysis.csv`
* **Ausgabe:** `results/reports/outlier_review.md`
* **Befehl:**
  ```bash
  .venv/bin/python scripts/visualization/generate_review_report.py
  ```

### `scripts/visualization/visualize_analysis.py`
Erstellt Plots zur Ähnlichkeitsverteilung, Kontextvergleichen (128 vs 512 vs 8192), NER-Recalls und POS-Verteilungen.
* **Befehl:**
  ```bash
  .venv/bin/python scripts/visualization/visualize_analysis.py --plots all
  ```

### `scripts/visualization/visualize_readability.py`
Plottet die Lesbarkeitswerte (Flesch, Wiener Sachtextformel, LIX) als Boxplots und Violinen-Diagramme und generiert das Dokument `research/readability_summary.md`.
* **Befehl:**
  ```bash
  .venv/bin/python scripts/visualization/visualize_readability.py
  ```

### `scripts/visualization/visualize_ttr.py`
Plottet die lexikalische Vielfalt (MATTR) nach Quellen und die Korrelation zwischen TTR und Textlänge.
* **Befehl:**
  ```bash
  .venv/bin/python scripts/visualization/visualize_ttr.py
  ```
