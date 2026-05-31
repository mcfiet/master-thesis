# 10 Dataset Analysis: Finaler Korpus-Review

## 1. Einleitung und Datensatz-Übersicht
Nach der Konsolidierung des Korpus und einer vollständigen Neuberechnung aller Metriken mit dem Jina-Modell (8192 Tokens Kontext) liegt nun die finale Analyse vor. Dieser Bericht dokumentiert den Prozess der Qualitätssicherung, bei dem der Rohdatenbestand (inkl. der neuen Quellen Hannover und TAZ) gefiltert wurde, um eine optimale Basis für das Modelltraining zu schaffen.

### 1.1 Korpus-Statistiken (Vergleich)

| Metrik | Roher Korpus | Bereinigter Korpus (Final) |
| :--- | :--- | :--- |
| **Anzahl Artikelpaare** | 1.533 | **1.476** |
| **Tokens Gesamt (AS)** | 1.868.994 | **1.787.327** |
| **Tokens Gesamt (LS)** | 1.468.345 | **1.432.434** |
| **Ø Tokens pro Satz (AS)** | 15,2 | 15,2 |
| **Ø Tokens pro Satz (LS)** | 8,3 | 8,3 |

## 2. Semantische Ähnlichkeit & Kontextlänge

### 2.1 Jina-Modell (Context Comparison)
Der Einsatz des Jina-Modells mit 8192 Tokens Kontext zeigt eine deutliche Verbesserung der Ähnlichkeitsmessung gegenüber Modellen mit kürzerem Kontext (128/512). Dies bestätigt, dass bei langen Behördentexten (insb. Hannover) relevante Informationen am Textende durch herkömmliche Modelle abgeschnitten wurden.

![Jina Context Comparison](img/analysis_final_cleaned/jina_context_comparison.png)

### 2.2 Semantische Ähnlichkeit nach Quelle (Bereinigt)
Im bereinigten Korpus wurden Paare mit einer Ähnlichkeit unter 0.60 entfernt. Dies stabilisiert die Qualität für das Modelltraining erheblich.

![Semantic Similarity by Source](img/analysis_final_cleaned/semantic_similarity_by_source.png)

## 3. Faktenerhalt & Faktentreue (NER)
Die bidirektionale NER-Analyse des finalen Sets zeigt eine hohe Konsistenz der Eigennamen zwischen AS und LS, was auf eine gute inhaltliche Ausrichtung der Paare hindeutet.

![Bidirectional NER](img/analysis_final_cleaned/bidirectional_ner_comparison.png)

## 4. Linguistische Analyse (Finales Set)
Die Analyse der Wortarten und Satzstrukturen bestätigt die Einhaltung der Regeln für Leichte Sprache auch im bereinigten Zustand.

### 4.1 Satzlängen-Vergleich
![Sentence Length](img/analysis_final_cleaned/sentence_length_comparison_bar.png)

### 4.2 Wortarten-Verteilung (POS)
![POS Distribution](img/analysis_final_cleaned/pos_distribution_bar.png)

## 5. Korpus-Bereinigung & Filterung
Basierend auf der initialen Analyse wurden Filterkriterien definiert, um die Qualität für das Modelltraining zu sichern.

### 5.1 Filterkriterien (Jina 8192)
- **Untergrenze Ähnlichkeit:** 0.60 (Entfernung von thematischen Fehltreffern)
- **Obergrenze Ähnlichkeit:** 0.99 (Entfernung von Fast-Duplikaten/Kopien)
- **Mindestlänge (LS):** 10 Tokens (Entfernung von leeren Teasern)
- **Inhalt:** Ausschluss von "Lorem Ipsum" Platzhaltern

### 5.2 Effekt der Bereinigung
Durch die Bereinigung wurden **55 Artikelpaare** entfernt. Dies reduziert das Rauschen im Datensatz (v.a. Teaser-Leichen), ohne die wertvolle Domänen-Abdeckung (z.B. Hannover mit über 750 Artikeln) nennenswert zu verringern.

## 6. Fazit
Der finale Datensatz (Version 3) umfasst **1.471 validierte Paare** mit insgesamt ca. **1,88 Mio. Tokens**. Die konsequente Nutzung von Long-Context Embeddings (Jina 8192) zur Validierung stellt sicher, dass die Alignments auch bei sehr langen Quelltexten inhaltlich korrekt sind. Damit liegt eine hochwertige Datenbasis für das Training von Simplification-Modellen vor.
