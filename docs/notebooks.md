# Notebook-Referenz (Jupyter Notebooks)

Diese Dokumentation beschreibt die Jupyter Notebooks im Ordner `notebooks/` sowie deren Rolle bei der Modellevaluierung und dem Training von Klassifikatoren.

---

## Übersicht der Notebooks

### 1. Satz-Klassifikator trainieren (`1_train_sentence_model.ipynb`)
Dieses Notebook trainiert ein rekurrentes neuronales Netz (Bidirectional LSTM) auf **Satzebene**, um zwischen alltagssprachlichen (AS) und leichtsprachlichen (LS) Sätzen zu klassifizieren.
* **Wesentliche Schritte:**
  1. **Daten laden & filtern:** Liest `data/analysis/information_loss_analysis_cleaned.csv` und filtert nach einem konfigurierbaren Bereich semantischer Ähnlichkeit (z.B. $0.8 \leq \text{Similarity} \leq 0.98$), um qualitativ schlechte Alignments auszuschließen.
  2. **Tokenisierung & Satzsplitting:** Zerlegt Artikel mittels SpaCy in Sätze und extrahiert Wörter (unter Filterung von Satzlängen $< 3$).
  3. **Klassen-Balancierung:** Da LS-Texte oft in deutlich mehr Sätze aufgeteilt sind als AS-Texte, führt das Notebook ein Random Under-Sampling auf Satzebene durch, um eine 50:50-Verteilung zu erzielen.
  4. **Modellarchitektur:** Definiert ein `BiLSTMClassifier` mit einem Embedding-Layer, einem bidirektionalen LSTM-Layer, Dropout ($0.3$) und einem finalen fully-connected Layer zur binären Klassifikation.
  5. **Training & Validierung:** Trainiert das Modell über 20 Epochen unter Speicherung des besten Validierungs-Modells (`results/models/lstm_baseline_sim_0.80_to_0.98.pt`).
  6. **Metriken:** Berechnet Genauigkeit, Balanced Accuracy, F1-Score und gibt eine Konfusionsmatrix aus.

### 2. Artikel-Klassifikator trainieren (`2_train_article_model.ipynb`)
Dieses Notebook trainiert ein BiLSTM auf **Artikelebene**, um ganze Texte als Alltagssprache oder Leichte Sprache zu klassifizieren.
* **Wesentliche Unterschiede zum Satz-Klassifikator:**
  - Höheres Sequenzlimit (`MAX_SEQ_LEN = 512` statt 100), um ganze Artikel abzudecken.
  - Kleinere Batch-Größe (`BATCH_SIZE = 32` statt 64) zur Vermeidung von Out-of-Memory-Fehlern.
  - Das Modell wird über 30 Epochen trainiert und speichert das Ergebnis als `results/models/lstm_article_sim_0.80_to_0.98.pt`.
  - Untersucht die Leistung bei der Vorhersage ganzer Textblöcke.

### 3. Mixup-Dataloader & Verteilungstest (`3_mixup_dataloader_test.ipynb`)
Dieses Notebook dient dem Testen und Visualisieren eines **Mixup-Verfahrens** für Textdaten zur Generierung kontinuierlicher Komplexitätswerte.
* **Hintergrund:**
  - Für Regressionsmodelle werden Texte mit kontinuierlichen Schwierigkeitsgraden (z. B. $0.25, 0.5, 0.75$) benötigt.
  - Neben LLM-generierten Texten kann dies über Mixup (Mischung von Satzstücken aus LS und AS) realisiert werden.
* **Funktionsweise des Mixup-Datasets:**
  - Schneidet zusammenhängende Slices (Abschnitte) aus den LS- und AS-Artikeln aus und fügt diese zusammen.
  - Der Start- und Endbereich der Slices wird stochastisch und unabhängig gewählt.
  - Die Ziel-Komplexität (Target) wird dynamisch berechnet als Verhältnis der Zeichenanzahl des LS-Anteils zur Gesamtanzahl an Zeichen im gemischten Absatz.
  - Das Notebook visualisiert die resultierenden Ziel-Verteilungen (Histogramme) für verschiedene Misch-Varianten, um sicherzustellen, dass die Verteilung der Trainingswerte gleichmäßig abgedeckt ist.
* **Notebook-Varianten für MixUp-Regression:**
  - `3_mixup_dataloader_test.ipynb`: Variante A (Statisch prä-generierte Mischungen).
  - `3_mixup_dataloader_test_getitem.ipynb`: Variante B (Rein dynamische Generierung on-the-fly in `__getitem__`).
  - `3_mixup_dataloader_test_getitem_cyclic.ipynb`: **Variante B (Dynamisch + Cyclic LR)** – Verwendet den rein dynamischen `__getitem__`-Dataloader mit einem `CosineAnnealingWarmRestarts` Learning-Rate-Scheduler zur Vermeidung lokaler Minima.
  - `3_mixup_dataloader_test_hybrid.ipynb`: Variante C (Hybrid aus statischen & dynamischen Samples).
  - `3_mixup_dataloader_test_hybrid_cyclic.ipynb`: Variante D (Hybrid + Cyclic LR).

### 4. Length-Bias-Überprüfung (`check_length_bias.ipynb`)
Dieses Notebook führt empirische Experimente durch, um zu prüfen, ob der Artikel-Klassifikator echte linguistische Strukturen lernt oder lediglich auf die Textlänge (Längen-Shortcut / Padding-Bias) optimiert.
* **Durchgeführte Experimente:**
  1. **Korrelationsanalyse:** Berechnet den Pearson- und Spearman-Korrelationskoeffizienten zwischen der Textlänge und der vom Modell vorhergesagten Wahrscheinlichkeit für Leichte Sprache. Ein hoher Korrelationskoeffizient deutet auf einen starken Längen-Bias hin.
  2. **Dummy-Text-Experiment (Konstanter Token-Test):** Ersetzt alle tatsächlichen Wörter im Testdatensatz durch einen Punkt `.`, während die ursprüngliche Satzlänge und die Padding-Nullen erhalten bleiben. Wenn das Modell diese inhaltsfreien Texte immer noch mit hoher Genauigkeit klassifizieren kann, lernt es primär über Padding und Länge.
  3. **Festlängen-Evaluation:** Schneidet alle alltagssprachlichen und leichtsprachlichen Texte starr auf dieselbe maximale Länge (z. B. 100 Token) ab und füllt kürzere Texte mit Padding auf. Das Modell wird auf diesen Texten evaluiert, um festzustellen, ob die Performance ohne Längenunterschiede einbricht.
* **Bedeutung:** Dieses Notebook ist essenziell für die methodische Validierung des Klassifikators, um "Shortcut Learning" auszuschließen.
