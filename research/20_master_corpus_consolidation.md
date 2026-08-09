# Master-Korpus: Skript-Konsolidierung & Diagnostik (Woche 20)

Dieses Dokument dokumentiert die Konsolidierung der verschiedenen Datenaufbereitungs- und Analysemethoden zu einer vereinheitlichten Master-Pipeline sowie die Durchführung von Korpus-Diagnosen und Boilerplate-Analysen vor und nach der Bereinigung.

---

## 1. Konsolidierung zum Master-Skript

Zuvor waren die Extraktion, Metrikberechnung und das Zusammenführen von Datensätzen über viele separate Skripte verstreut. Dies wurde in einem zentralen Master-Skript konsolidiert.

### Features des Master-Skripts:
* **Vereinheitlichtes Daten-Alignment:** Lädt bereinigte Quellpaare (AS/LS) und überführt sie in ein zentrales Datenformat.
* **Semantische Ähnlichkeit:** Nutzt ein Long-Context-Sprachmodell (Jina Embeddings) mit einem vollen Kontextfenster von 8192 Tokens, um die semantische Übereinstimmung via Kosinus-Ähnlichkeit präzise zu messen.
* **NER-Recall (Named Entity Recognition):** Berechnet mithilfe eines SpaCy-Modells den Entity-Recall in beide Richtungen (AS -> LS zur Prüfung des Informationsverlusts; LS -> AS zur Überprüfung von Halluzinationen).
* **Lesbarkeitsmetriken:** Automatische Berechnung von Flesch Reading Ease (für Deutsch), Wiener Sachtextformel (WSTF) und LIX.
* **Lexikalische Diversität:** Berechnung der Type-Token-Ratio (TTR) sowie Moving Average Type-Token-Ratio (MATTR) mit konfigurierbarem Fenster (Standard: 50 Wörter), um Längeneffekte zu eliminieren.

---

## 2. Boilerplate-Analyse und Bereinigung

Es wurde eine umfassende Analyse der Boilerplate-Texte über die verschiedenen Quellen hinweg durchgeführt. 
* **Vor der Bereinigung:** In bestimmten Quellen (wie Apotheken, Köln oder Stuttgart) wiesen bis zu 100% der Sätze wiederkehrende, inhaltliche leere Phrasen (wie „Mehr Informationen“, „Achtung/Wichtig“, „Prüfung/Umsatz“) auf, was die Regressoren verzerren konnte.
* **Nach der Bereinigung:** Durch gezielte Filterung der Boilerplate-Klassen konnte der Anteil unerwünschter Textsegmente drastisch gesenkt werden, um eine saubere Datenbasis für die Modellierung zu garantieren.

---

## 3. Korpus-Diagnostik

Auf Basis des generierten Master-Korpus wurden statistische Auswertungen durchgeführt:
* **Datensatz-Umfang:** Analyse der Wort- und Satzlängenverteilungen zwischen Alltagssprache (AS) und Leichter Sprache (LS).
* **Ergebnisse:** LS-Texte weisen eine signifikant geringere lexikalische Dichte und stark verkürzte Sätze auf. In über 80% der Fälle bleiben die zentralen Eigennamen (NER-Recall) erhalten, was die hohe Qualität des Alignments bestätigt.
