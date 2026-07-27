# Zusammenfassender Statusbericht & Planung: Master Thesis

**Titel:** Automatische Übersetzung in Leichte Sprache  
**Autor:** Fiete Scheel  
**Stand der Arbeit:** Juli 2026  

---

## 1. Übersicht & Gesamtarchitektur

Das Ziel dieser Masterarbeit ist die Entwicklung eines integrierten, qualitätsgesicherten Systems zur automatischen Übersetzung von Alltagssprache (AS) in Leichte Sprache (LS). Das Gesamtprojekt gliedert sich in eine **Drei-Stufen-Pipeline**:

```
[ Step 1: Datenkorpus ] ──► [ Step 2: Metrik- & Regressionsmodell ] ──► [ Step 3: Übersetzungsmodell ]
  • Multi-Quellen Crawling    • SBERT / BiLSTM Klassifikation           • Generierung (Seq2Seq / LLM)
  • Qualitatives Alignment    • Continuous MixUp Regression             • Reward-Guided Training (RLHF/DPO)
  • Multi-Level Cleaning      • Out-of-Domain & Bias Evaluation         • Evaluation (Faktentreue & Stil)
```

---

## 2. Detaillierter Stand der bisherigen Arbeit

### 2.1 Step 1: Datenbasis & Korpus-Erstellung

Die Datenbasis bildet das Fundament der gesamten Arbeit. Es wurde ein umfangreiches, qualitativ bereinigtes Textkorpus aus 11 verschiedenen öffentlichen und redaktionellen Quellen aufgebaut.

#### Quellabdeckung & Korpus-Statistiken (Bereinigter Finaler Stand):
* **Gesamtumfang:** 1.471 verifizierte, parallel ausgerichtete Artikel-Paare.
* **Token-Umfang:** ~1,43 Millionen LS-Tokens vs. ~1,78 Millionen AS-Tokens (gemessen mit `tiktoken cl100k_base`).
* **Satz-Umfang:** 91.103 LS-Sätze vs. 54.256 AS-Sätze.
* **Durchschnittliche Satzlänge:** 8,3 Wörter/Satz (LS) vs. 15,2 Wörter/Satz (AS).

| Quelle | Paare | Tokens (LS) | Tokens (AS) | W/S (LS) | W/S (AS) | Hauptmerkmale & Reinigungsschritte |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Hannover.de** | 796 | 861.967 | 858.086 | 8.0 | 15.6 | Größte Einzelquelle; behördliche & redaktionelle Texte (§ 9a NBGG). Rekursiv gecrawlt. |
| **MDR** | 227 | 94.976 | 168.440 | 9.6 | 14.7 | Nachrichtenbeiträge. Entfernung von "Echo-Effekten" (doppelte P-Tags) & Radio-Metadaten. |
| **Apotheken Umschau** | 157 | 234.608 | 443.722 | 7.8 | 13.6 | Medizinische Fachtexte. Entfernung von Bildbeschreibungen, TOCs & Copyright-Vermerken. |
| **Hamburg.de** | 56 | 61.204 | 66.977 | 9.3 | 15.7 | Kommunalverwaltung. Striktes Sprachleisten-Alignment; MT-disclaimer-gefiltert. |
| **Behindertenbeauftragter** | 51 | 38.724 | 44.725 | 9.3 | 19.0 | Politische Rahmenbedingungen. Entfernung von PDF-Stubs & Fallback-URLs. |
| **Wiesbaden.de** | 41 | 13.808 | 23.332 | 8.7 | 19.1 | Kommunale Artikel. Gecrawlt & filtered; mäßiges Alignment auf CMS-Ebene dokumentiert. |
| **Stuttgart.de** | 39 | 42.894 | 91.331 | 10.1 | 17.0 | Behördenservice. Parameter-basierte URL-Zuordnung; Eliminierung von Sharing-Tools. |
| **Stadt Köln** | 38 | 55.985 | 43.275 | 9.2 | 16.1 | Wayback-Machine Archiv. Encoding-Autodetect & Parent-Deduplizierung. |
| **Lebenshilfe Main-Taunus** | 34 | 9.974 | 11.037 | 8.6 | 18.3 | Platzhalter-Filterung, Content-Hashing & Truncation von Kontakt-Blöcken. |
| **Brand Eins** | 18 | 6.044 | 7.216 | 32.1 | 18.2 | Journalismus. Deep-Color-Inspection for inline HTML LS/AS-Splitting. |
| **Sozialpolitik.com** | 14 | 9.249 | 23.573 | 9.4 | 15.6 | Politische Bildung. Sauberes manuelles Alignment. |
| **GESAMT (Final)** | **1.471** | **1.429.433** | **1.781.714** | **8.3** | **15.2** | **Bereinigter Sweet-Spot Datensatz** |

#### Methodik der Datenbereinigung & Quality Assurance:
1. **Long-Context Embedding Alignment:** Einsatz von `jina-embeddings-v2-base-de` (8.192 Tokens Kontext) zur Vermeidung von Informationsabschnitt-Fehlern bei langen behördlichen Dokumenten.
2. **Similarity Sweet-Spot Filtering:** Selektion des optimalen Schwellenwertbereichs von $0.80 \le \text{Similarity} \le 0.98$, um unzusammenhängende Paare ($<0.60$) sowie Duplikate ($>0.99$) auszuschließen.
3. **Automatisierte Nachbereinigung (Post-Cleaning):** Normalisierung des Mediopunkts (`·`) zur Silbentrennung, Entfernung von Boilerplate-Metadaten (Radio-Sendezeiten, Autorenzeilen) und Reparatur fehlender Leerzeichen nach Satzzeichen.

---

### 2.2 Step 2: Metrik- & Regressionsmodellierung

Um die Qualität von Leichter Sprache objektiv und automatisiert zu bewerten, wurden sowohl **Klassifikations-** als auch kontinuierliche **Regressionsmodelle** entwickelt und evaluiert.

#### 1. Baseline BiLSTM Klassifikator & Out-of-Domain Generalisierung:
* **Trainingsdaten:** Gecrawlter Web-Korpus.
* **Evaluierungsebenen:** Gegenüberstellung von direktem Artikel-Training vs. Satzebene mit Aggregation (Majority Voting).

##### Evaluierungsergebnisse der Klassifikationsmodelle:

| Modell-Ansatz | Evaluierungs-Ebene | In-Domain Test BAcc | Out-of-Domain (LH) BAcc | Out-of-Domain Perfect Pair Match |
| :--- | :--- | :---: | :---: | :---: |
| **Artikel-Level Modell** | Dokument / Artikel | 95,92% | 90,82% | 81,63% (40/49) |
| **Satz-Level Modell** | Einzelne Sätze | 92,00% | 77,74% | – |
| **Satz-Level Modell (Majority Vote)** | Aggregiert auf Artikel | **99,68%** | **97,96%** | **95,92% (47/49)** |

* **Empirische Bias-Kontrollexperimente:**
  * **Absatz-/Layout-Bias:** Kontrolltest ohne Zeilenumbrüche ergab 100% identische Konfidenzwerte, da Whitespace im Tokenizer verworfen wird.
  * **Längen-Bias:** Korrelation zwischen Textlänge und Vorhersagewahrscheinlichkeit ist vernachlässigbar ($r = 0.1730$, $p > 0.05$). Dummy-Text-Test (Ersetzung der Wörter durch `.`) führte zum Zusammenbruch der Accuracy auf exakt $50,00\%$. Slicing auf exakt 100 Tokens hielt eine Balanced Accuracy von $87,76\%$. Längen-Overfitting ist somit empirisch ausgeschlossen!

#### 2. Continuous MixUp Regression (Bewertung von Sprachkomplexitäts-Gradienten):
Zur Messung fließender Übergänge zwischen Alltagssprache ($\lambda = 0.0$) und Leichter Sprache ($\lambda = 1.0$) wurden synthetische Satzmischungen erzeugt und vier Modellvarianten des `BiLSTMRegressor` verglichen.

##### Systematischer Modellvergleich (MixUp-Varianten):
* **Variante A (Statisch):** Feste Satzmischungen im Konstruktor (`3_mixup_dataloader_test.ipynb`). Stabil, aber Overfitting-gefährdet (Test MSE: 0,0383, Test Acc: 91,55%).
* **Variante B (Dynamisch):** Dynamisches On-the-Fly Shuffling in `__getitem__` (`3_mixup_dataloader_test_getitem.ipynb`). Hohe Varianz, schlechte Konvergenz, bimodale Vorhersage-Cluster (Test MSE: 0,0758, Test Acc: 80,17%).
* **Variante B + Cyclic LR:** Zusatz von `CosineAnnealingWarmRestarts` (`3_mixup_dataloader_test_getitem_cyclic.ipynb`). Verringert den Val-Loss punktuell auf 0,0505, korrigiert jedoch nicht das Fehlen fester Ankerpunkte.
* **Variante C (Hybrid):** Mischerverhältnis von statischen zu dynamischen Samples steigt von 0% auf 100% über die Epochen (`3_mixup_dataloader_test_hybrid.ipynb`). Starke Performance (Test MSE: 0,0267, Test Acc: 95,04%).
* **Variante D (Hybrid + Cyclic LR):** Kombination aus Hybrid-Dataloader und zyklischer Lernrate (`3_mixup_dataloader_test_hybrid_cyclic.ipynb`). **Top-Performer über alle Metriken!**

##### Evaluierungsergebnisse der Regressionsmodelle:

| Modell-Variante | In-Domain Test MSE | In-Domain Test MAE | In-Domain Test Acc | Out-of-Domain (LH) BAcc | Out-of-Domain (LH) MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Variante A (Statisch)** | 0.0383 | 0.1557 | 91.55% | 85.75% | 0.2888 |
| **Variante B (Dynamisch)** | 0.0758 | 0.2264 | 80.17% | 86.32% | 0.3378 |
| **Variante C (Hybrid)** | 0.0267 | 0.1158 | 95.04% | 85.10% | 0.2695 |
| **Variante D (Hybrid + Cyclic)** | **0.0241** | **0.1027** | **95.92%** | **89.37%** | **0.2036** |

##### Visualisierung & Verteilungsanalyse:
Die Auswertung mittels 2D-Histogrammen, Scatterplots und KDE-Dichteverteilungen zeigt für **Variante D**:
* **Gegenüber Alltagssprache ($\lambda = 0.0$):** Extrem scharfer Peak bei $\bar{\lambda}_{AS} = 0,0957$ (hohe Sicherheit durch Komplexitätsmarker wie Passiv und Fachwörter).
* **Gegenüber Leichter Sprache ($\lambda = 1.0$):** Breitere Verteilung um $\bar{\lambda}_{LS} = 0,7293$ (da Leichte Sprache primär durch das Fehlen von Komplexität definiert ist).

---

## 3. Offene Punkte & Verbleibender Arbeitsplan

Um die Masterarbeit erfolgreich abzuschließen, stehen folgende konkrete Schritte an:

### Phase 1: Synthetische LLM-Stufen Evaluierung (Kurzfristig)
1. **Validierung der synthetischen LLM-Stufen:**
   * Auswertung des trainierten Regressors (Variante D) auf den mit Ollama/FlensGen GPT (`generate_synthetic_regression_steps.py`) erzeugten Zwischenstufen ($0.25$, $0.50$, $0.75$) des Lebenshilfe-Sets.
   * Überprüfung der Monotonie: Korrelieren die Modellvorhersagen mit den vom LLM geforderten Ziel-Stufen?

### Phase 2: Step 3 – Konzeption & Training des Übersetzungsmodells (Hauptfokus)
1. **Datenaufbereitung für die Übersetzungsaufgabe:**
   * **Auflösung des 1:n-Problems:** Gegenüberstellung von reinem Satz-Alignment vs. **Block-/Absatz-Alignment**, um Kontext und Information zu wahren.
   * Bereitstellung von Trainings-Splits (Train / Val / Test) auf Basis der Similarity-Filterung.
2. **Modellauswahl & Baseline Supervised Fine-Tuning (SFT):**
   * **Encoder-Decoder Modellauswahl:** mBART, mt5.
   * **Decoder-Only LLM Fine-Tuning (LoRA / QLoRA):** LLaMA-3 (8B), Mistral (7B).
3. **Reward-Guided Fine-Tuning / RLHF / DPO:**
   * **Nutzung des Metrik-Modells als Reward:** Integration unseres best-performenden MixUp-Regressors (Variante D) bzw. SBERT-Klassifikators als implizite Reward-Funktion im Reinforcement Learning (PPO / DPO), um Halluzinationen zu unterdrücken und die Formalien der Leichten Sprache zu erzwingen.

### Phase 3: Evaluierung & Thesis-Synthese
1. **Mehrdimensionale Evaluierung der Generate-Outputs:**
   * **Stil- & Regelkonformität:** Score des MixUp-Regressors & klassischen Lesbarkeitsindizes (Flesch, Wiener).
   * **Inhaltliche Faktentreue & Information Loss:** Bidirektionale Named Entity Recognition (NER) / NLI Entailment Check zur Messung von Informationsverlust vs. Halluzination.
2. **Schreiben der Thesis:**
   * Strukturierung nach den in `thesis_questions.md` ausgearbeiteten Forschungsfragen (FF 2.1–2.5, FF 3.1–3.3, FF 4.1–4.5).

---

## 4. Relevante Dateien & Skripte im Repository

* **Forschungsnotizen & Visualisierungen:**
  * Status-Quo Overview: [00_thesis-status-quo.md](file:///Users/fietescheel/Documents/Master%20Thesis/research/00_thesis-status-quo.md)
  * Datensatz & Cleaning: [05_data_corpus.md](file:///Users/fietescheel/Documents/Master%20Thesis/research/05_data_corpus.md), [10_dataset_analysis_final.md](file:///Users/fietescheel/Documents/Master%20Thesis/research/10_dataset_analysis_final.md)
  * Klassifikation & Biases: [13_model_training.md](file:///Users/fietescheel/Documents/Master%20Thesis/research/13_model_training.md)
  * Regressions- & MixUp-Performance: [17_regression_performance.md](file:///Users/fietescheel/Documents/Master%20Thesis/research/17_regression_performance.md), [18_regression_optimization.md](file:///Users/fietescheel/Documents/Master%20Thesis/research/18_regression_optimization.md)
  * Forschungsfragen: [thesis_questions.md](file:///Users/fietescheel/Documents/Master%20Thesis/research/thesis_questions.md)
* **Notebooks:**
  * Baseline Training: [1_train_sentence_model.ipynb](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/1_train_sentence_model.ipynb)
  * MixUp Regressor Varianten: [3_mixup_dataloader_test_hybrid_cyclic.ipynb](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/3_mixup_dataloader_test_hybrid_cyclic.ipynb)
  * MixUp Modell-Evaluation: [4_mixup_model_evaluation.ipynb](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/4_mixup_model_evaluation.ipynb)
