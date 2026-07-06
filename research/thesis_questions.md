# Forschungsfragen und Modellvergleiche (Thesis)

Dieses Dokument fasst die Forschungsfragen, Unterfragen und empirischen Vergleichsmöglichkeiten zusammen, die sich aus dem bisherigen Verlauf der Arbeit (Datenbereinigung, Metrikenanalyse, Modelltraining und Out-of-Domain-Evaluation) ergeben haben.

---

## 1. Übergreifende Leitfrage (Hauptfrage)
*Wie lässt sich die automatische Übersetzung in Leichte Sprache (LS) durch die Kombination aus semantisch optimierten Korpora, strukturellen Klassifikatoren und zielgerichteten Trainingsstrategien systematisch verbessern und evaluieren?*

---

## 2. Forschungsfragen zur Datenbasis & Alignment (Corpus)

*   **FF 2.1: Block-basiertes vs. Satz-basiertes Alignment (Das 1:n-Problem)**
    *   *Unterfrage:* Wie stark beeinträchtigt das 1:n-Satzalignment (ein komplexer Satz der Alltagssprache wird zu mehreren einfachen Sätzen in LS) das Training von Sequenz-zu-Sequenz-Modellen?
    *   *Unterfrage:* Kann ein block- bzw. absatzweises Alignment semantische Einheiten besser wahren und den Informationsverlust im Vergleich zu reinem Satzalignment verringern?
*   **FF 2.2: Der qualitative "Sweet Spot" der semantischen Ähnlichkeit**
    *   *Unterfrage:* Welcher Filterbereich der Kosinus-Ähnlichkeit (z. B. 0.60–0.98, 0.70–0.98 oder 0.80–0.98) bei parallelen Textpaaren liefert die qualitativ beste Grundlage für das Modelltraining?
    *   *Unterfrage:* Wie verhält sich die Balance zwischen Datenmenge (Sample-Anzahl) und der semantischen Konsistenz der Zieltexte?
*   **FF 2.3: Einfluss der Kontextfenstergröße bei der Filterung**
    *   *Unterfrage:* Inwieweit verbessern Long-Context Embeddings (z. B. Jina-embeddings-v2 mit 8.192 Tokens) die Ähnlichkeitsbestimmung und die Rauschreduktion bei langen behördlichen Texten im Vergleich zu Standard-Embeddings (z. B. MiniLM mit 128/512 Tokens)?
*   **FF 2.4: Lexikalische Diversität und Quellenspezifika**
    *   *Unterfrage:* Wie stark variiert die lexikalische Diversität (Type-Token-Ratio / MATTR) zwischen verschiedenen Textquellen (z. B. journalistische Texte wie bei der *TAZ* vs. administrative Webseiten) und wie spiegelt sich dies im Klassifikationsverhalten wider?
*   **FF 2.5: Auswirkungen von systematischem Datenrauschen**
    *   *Unterfrage:* Welchen Einfluss haben Scraping-Artefakte (z. B. Metadaten-Konkatenation bei *Brand Eins*, Standard-Footer beim *MDR*) auf das Generierungsverhalten von LLMs und wie stark trägt eine automatisierte Nachbereinigung (Post-Cleaning) zur Modellgüte bei?

---

## 3. Forschungsfragen zur Klassifikation, Generalisierung & Biases

*   **FF 3.1: Satzebene vs. Dokumentenebene**
    *   *Unterfrage:* Ist Leichte Sprache ein lokales (satzweises) oder ein holistisches (dokumentenweites) Phänomen?
    *   *Unterfrage:* Warum steigt die Klassifikationsgenauigkeit beim Wechsel von Satz- zu Artikelebene so drastisch an (von ~93% auf ~99% BAcc)?
    *   *Unterfrage:* Wie effektiv ist die Aggregation von Satzvorhersagen mittels Mehrheitsentscheidung (Majority Voting) für die Dokumentenklassifikation?
*   **FF 3.2: Generalisierungsfähigkeit (Out-of-Domain-Evaluation)**
    *   *Unterfrage:* Wie gut generalisieren Klassifikatoren, die auf gecrawlten Web-Korpora trainiert wurden, auf komplett unabhängige, proprietäre Dokumente (z. B. den unveröffentlichten Lebenshilfe-Datensatz) ohne zusätzliches Fine-Tuning?
*   **FF 3.3: Empirischer Ausschluss von Klassifikations-Shortcuts (Biases)**
    *   *Unterfrage (Length-Bias):* Lernt das Modell primär den Längenunterschied (oder den Anteil an Padding-Nullen im Vektor) als Abkürzung?
    *   *Unterfrage (Layout-Bias):* Nutzt das Modell Absatzstrukturen und Zeilenumbrüche als Feature und wie kann dies durch Tokenisierung (Entfernen von Whitespace) verhindert werden?
    *   *Unterfrage (Typografie-Bias):* Inwiefern dienen typografische Besonderheiten Leichter Sprache (z. B. der Mediopunkt `·` zur Silbentrennung) als Shortcut für das Modell und wie beeinflusst ihre Normalisierung die Repräsentationsfähigkeit?

---

## 4. Konkrete Vergleichsszenarien (Was du vergleichen kannst)

Hier sind die verschiedenen Ansätze und Modelle aufgelistet, die du trainiert hast oder noch trainieren kannst, um sie in der Arbeit gegenüberzustellen:

### A. Modellarchitektur & -komplexität
*   **BiLSTM vs. Sentence-BERT / Transformer-Encoder:**
    *   Vergleich der Klassifikationsleistung (Balanced Accuracy, F1-Score) und der Trainingseffizienz (Laufzeit, Ressourcen).
    *   *Erkenntnis aus deinen Notizen:* Die BiLSTM-Baseline schlägt eingefrorene SBERTs und erzielt auf Artikelebene 99% BAcc bzw. out-of-domain 97.96% via Majority Vote.
*   **Context Window (128 vs. 512 vs. 8.192 Tokens):**
    *   Vergleich von `MiniLM-L12-v2` (bis zu 512 Tokens) mit `jina-embeddings-v2-base-de` (bis zu 8.192 Tokens).
    *   Wie stabil bleiben die Ähnlichkeitsscores bei steigender Dokumentenlänge?

### B. Trainingsparadigmen (SBERT Fine-Tuning-Varianten)
Vergleich der vier im `genai-project` getesteten Konfigurationen hinsichtlich Performance und Ressourcenbedarf:
1.  **Vollständiges Fine-Tuning (End-to-End):** Höchste Genauigkeit (ca. 0.91–0.95 BAcc), schnelle Konvergenz.
2.  **Sentence-BERT + LoRA (Parameter-effizient):** Sehr nah am Voll-Tuning (ca. 0.90 BAcc), spart Grafikspeicher.
3.  **Nur letzte Schicht frei (Frozen Backbone):** Mäßige Performance (~0.83 BAcc).
4.  **Komplett eingefroren (Feature Extractor + MLP/linearer Kopf):** Ungenügende Performance (0.71–0.76 BAcc).

### C. Einfluss der Datenfilterung (Similarity-Schwellenwerte)
*   Vergleiche das Training desselben Klassifikators (z. B. BiLSTM) auf Datensätzen, die mit unterschiedlichen minimalen Ähnlichkeiten gefiltert wurden:
    *   **Baseline (Roher Korpus / ungefiltert)**
    *   **Set A (Similarity 0.60 - 0.98)**
    *   **Set B (Similarity 0.70 - 0.98)**
    *   **Set C (Similarity 0.80 - 0.98) -- dein aktueller Sweet Spot**
    *   **Set D (Similarity 0.90 - 0.98)**
    *   *Frage:* Führt eine strengere Filterung trotz geringerer Datenmenge zu einer besseren Out-of-Domain-Generalisierung?

### D. Vokabular-Pruning (Umgang mit seltenen Wörtern)
*   **Standard-Vokabular vs. Pruned-Vokabular (< 3 Vorkommen maskiert):**
    *   *Szenario:* Seltene Wörter werden zu `<unk>` (Unknown) umgewandelt.
    *   *Hypothese zum Vergleich:* Verhindert das Pruning ein Overfitting auf domänenspezifisches Vokabular (z. B. JVA-Begriffe) und zwingt das Modell, sich auf syntaktische Strukturen zu konzentrieren? Oder wird das `<unk>`-Token selbst zum Shortcut ("viele Unbekannte = schwere Sprache")?

### E. Übersetzungsmodelle: Generierung & Evaluation (RLHF/RL-Ansatz)
*   **SFT vs. RLHF (Klassifikator als Reward):**
    *   Vergleich des Generierungsmodells (z. B. **mt5** oder **Mistral LoRA**), einmal trainiert via Supervised Fine-Tuning (SFT) und einmal optimiert mit Reinforcement Learning unter Nutzung des trainierten SBERT-Klassifikators als Reward-Funktion.
    *   Vergleich mittels klassischer Metriken (BLEU, ROUGE - obwohl für LS ungeeignet), linguistischer Formeln (Flesch, Wiener) und dem Reward-Modell selbst.

---

## 5. Zukünftige Fragen, die noch nicht bedacht wurden (Erweiterungspotenzial)

Diese Aspekte könnten die Diskussion deiner Thesis bereichern oder als Anknüpfungspunkte für "Future Work" dienen:

1.  **Faktentreue vs. Vereinfachung (Das Problem des Halluzinierens/Weglassens):**
    *   *Frage:* Wie lässt sich automatisiert unterscheiden, ob ein Informationsverlust eine *legitime Vereinfachung (Zusammenfassung)* ist oder ein *Verlust essenzieller Fakten*?
    *   *Ansatz:* Könnte man eine separate Metrik auf Basis von Named Entity Recognition (NER) oder Entailment-Modellen (NLI) nutzen, um die inhaltliche Abdeckung unabhängig von der stilistischen Einfachheit zu bewerten?
2.  **Gibt es "die" Leichte Sprache? (Domänenspezifik):**
    *   *Frage:* Unterscheiden sich die sprachlichen Muster von Leichter Sprache je nach Anwendungsgebiet?
    *   *Ansatz:* Verhalten sich Modelle anders, wenn sie auf medizinischen Texten (*Apotheken Umschau*), politischen Texten (*sozialpolitik.com*) oder behördlichen Texten (*Hannover.de*) evaluiert werden? Ist ein domänenspezifisches Training notwendig oder reicht ein allgemeiner LS-Klassifikator?
3.  **Die Rolle von LLM-Prompting im Vergleich zu kleineren, trainierten Modellen:**
    *   *Frage:* Können moderne Large Language Models (z. B. GPT-4o, Llama 3) mittels ausgefeiltem Few-Shot Prompting oder In-Context Learning bessere Übersetzungen in Leichte Sprache anfertigen als deine spezialisierten kleineren Modelle (mt5 / Mistral LoRA)? Wie verhalten sich beide Ansätze bezüglich syntaktischer Korrektheit (Regelkonformität der Leichten Sprache)?
4.  **Vom Binären Klassifikator zur kontinuierlichen Vereinfachungs-Regression:**
    *   *Frage:* Statt Texte nur binär in "Normal" und "Einfach" einzuteilen: Lässt sich der Vereinfachungsgrad auf einer kontinuierlichen Skala (z. B. 0.0 für schwere Fachsprache bis 1.0 für extreme Leichte Sprache) vorhersagen?
    *   *Ansatz:* Evaluierung der Mix-Up-Szenarien. Kann ein solches Regressionsmodell zur Qualitätskontrolle für menschliche Übersetzer genutzt werden (z. B. als Feedback-Tool)?
5.  **Zielgruppen-Validierung (Human Evaluation):**
    *   *Frage:* Inwieweit korrelieren die hohen Werte deines Klassifikators und die klassischen Lesbarkeitsmetriken (Flesch/Wiener) tatsächlich mit dem Leseverständnis und der kognitiven Entlastung der eigentlichen Zielgruppe (Menschen mit Lernschwierigkeiten, geringer Lese- und Schreibkompetenz oder Deutsch als Zweitsprache)?
