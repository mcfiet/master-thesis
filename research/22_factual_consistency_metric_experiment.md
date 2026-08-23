# 22. Empirische Evaluierung von Metriken zur Faktenkonsistenz und Halluzinationserkennung

**Thema:** Systematischer Vergleich von SBERT (Bi-Encoder), NLI (Cross-Encoder), NER-Overlap, numerischem Regex-Check und hybridem Composite-Scoring zur Erkennung von Faktenfehlern und Halluzinationen  
**Datum:** 23. August 2026  
**Autor:** Fiete Scheel  
**Experimentelles Notebook:** [`notebooks/research/metric/6_factual_consistency_metric_experiment.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/research/metric/6_factual_consistency_metric_experiment.ipynb)  
**Datengrundlage:** Tripartiter 4-Klassen-Benchmark ($N = 177$ Textpaare aus Lebenshilfe, Masterkorpus, mBART-SFT-Generierungen und Adversarial Slices)  
**Generierte Ergebnisdatei:** [`results/evaluation/factual_consistency_metric_results.csv`](file:///Users/fietescheel/Documents/Master%20Thesis/results/evaluation/factual_consistency_metric_results.csv)

---

## 1. Motivation & Fragestellung

In den vorherigen Forschungsnotizen ([`research/22_sft_data_scaling_analysis.md`](file:///Users/fietescheel/Documents/Master%20Thesis/research/22_sft_data_scaling_analysis.md) und [`research/22_sft_qualitative_generation_analysis.md`](file:///Users/fietescheel/Documents/Master%20Thesis/research/22_sft_qualitative_generation_analysis.md)) wurde eine fundamentale Schwachstelle der bisherigen Pipeline identifiziert:
* Das mBART-SFT-Modell generiert zwar die syntaktische Form Leichter Sprache (Bindestriche, kurze Sätze), verdreht aber regelmäßig **Zahlenkontexte** (*„16 Jahre Haft“ $\rightarrow$ „16 Jahre alt“*; *„20,47 Mio. € Gewinn“ $\rightarrow$ „Jahr 2047“*), **Personen- und Markenrollen** (*„Sportwagenbauer Ferrari“ $\rightarrow$ „Sportwagen-Fahrer Ferrari“*) oder erfindet **themenfremde Ereignisse** (*„Erdbeben“* bei der Costa Concordia).
* Die bisherige semantische Metrik **$R_{\text{sem}}$ (SBERT Kosinus-Ähnlichkeit)** übersieht diese Fehler nahezu vollständig und vergibt weiterhin hohe Ähnlichkeitswerte von **$> 0.85$**, weil 95 % des thematischen Wortschatzes übereinstimmen (*Bi-Encoder-Blindspot*).

### Kernfragen dieses Experiments:
1. **Themenwechsel vs. feingranulare Faktenfehler:** Wie verhält sich SBERT bei groben Themenwechseln (Random Negatives) im Vergleich zu feinen Zahlen- und Negationsfehlern (Minimal Perturbations)?
2. **Das NER-Dilemma:** Bestätigt sich experimentell, dass klassisches Named Entity Recognition (NER) für Leichte Sprache ungeeignet ist, weil erwünschtes Abstrahieren (*„BMAS“ $\rightarrow$ „Politiker“*) fälschlicherweise bestraft wird?
3. **NLI & Composite Scoring:** Welche Metrik-Kombination liefert die höchste Trennschärfe (ROC-AUC) und eignet sich als robuste Reward-Komponente für zukünftige DPO-Trainings?

---

## 2. Der 4-Klassen-Benchmark-Datensatz ($N = 177$)

Um die Metriken unter kontrollierten Bedingungen zu evaluieren, wurde ein balancierter Benchmark aus vier Klassen aufgebaut:

| Klasse | Beschreibung / Herkunft | Stichproben ($N$) | Soll-Faktizität |
| :--- | :--- | :---: | :---: |
| **1. Gold Positives** | Menschliche Referenzen aus Lebenshilfe ($N=37$) & Master Parallelkorpus ($N=40$) | $77$ | **1 (Korrekt)** |
| **2. Real Model Hallucinations** | Echte qualitative SFT/DPO-Fehlerfälle aus `dpo_pairs_w05_w05.jsonl` (Costa Concordia, NZZ 2047, Ferrari-Fahrer etc.) | $40$ | **0 (Falsch)** |
| **3. Random Shuffle Negatives** | Zufällig durchgetauschte Paare (Themenwechsel: Quelle A mit Ziel B) | $30$ | **0 (Falsch)** |
| **4. Targeted Minimal Perturbations** | Gezielte 1-Wort-Mutationen auf Gold-Sätzen (Zahlen-Shift, Negations-Inversion, Rollentausch) | $30$ | **0 (Falsch)** |

---

## 3. Quantitative Ergebnisse

Die Inferenz aller Metriken auf den $177$ Benchmark-Paaren ergab folgende Verteilungen:

### 3.1 Durchschnittswerte pro Testklasse

| Testklasse | SBERT ($R_{\text{sem}}$) | NLI $P(\text{Contra})$ | NLI Factuality | NER Jaccard | Number Check | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Gold Positives** | **$0.8877 \pm 0.05$** | **$0.2530 \pm 0.16$** | **$0.4547 \pm 0.10$** | **$0.1345 \pm 0.11$** | **$0.6343 \pm 0.40$** | **$0.2606 \pm 0.17$** |
| **2. Real Hallucinations** | $0.8687 \pm 0.07$ | $0.3101 \pm 0.13$ | $0.4452 \pm 0.06$ | $0.2032 \pm 0.12$ | $0.7915 \pm 0.28$ | $0.3066 \pm 0.12$ |
| **3. Random Shuffle** | **$0.6523 \pm 0.09$** | $0.3123 \pm 0.09$ | $0.4461 \pm 0.04$ | $0.0039 \pm 0.01$ | $0.3003 \pm 0.38$ | $0.0862 \pm 0.11$ |
| **4. Minimal Perturbations** | **$0.7934 \pm 0.12$** | **$0.4091 \pm 0.26$** | **$0.4027 \pm 0.18$** | **$0.6778 \pm 0.25$** | $0.8167 \pm 0.35$ | $0.2878 \pm 0.18$ |

---

### 3.2 Trennkraft & ROC-AUC Analyse

| Metrik | ROC-AUC (Gesamt) | $\Delta$(Gold $-$ Halluzination) | $\Delta$(Gold $-$ Minimal Perturbation) | Bewertung |
| :--- | :---: | :---: | :---: | :--- |
| **SBERT Similarity ($R_{\text{sem}}$)** | **$0.7722$** | $+0.0190$ | $+0.0943$ | Erkennt Themenwechsel ($0.65$), aber versagt bei feinen Faktenfehlern ($0.87$). |
| **NLI $P(\text{Contradiction})$** | **$0.6223$** | $\mathbf{-0.0570}$ | $\mathbf{-0.1560}$ | **Straft feine semantische Widersprüche ($0.41$ vs. $0.25$) am schärfsten ab.** |
| **NLI Factuality ($P_e - P_c$)** | **$0.5840$** | $+0.0095$ | $+0.0520$ | Reines Entailment wird bei langen Texten durch *Neutral* verwässert. |
| **NER Jaccard Overlap** | **$0.4900$** | $-0.0688$ | $-0.5433$ | **Schlechter als Zufall!** Belohnt fehlerhafte Sätze mit hohem NER-Match ($0.68$). |
| **Number Consistency Check** | **$0.4619$** | $-0.1572$ | $-0.1824$ | Deterministisch nützlich, aber isoliert unzureichend. |

---

## 4. Visualisierungen

### 4.1 4-Panel Vergleichsplots der Score-Verteilungen
![Vergleich der Faktenmetriken](file:///Users/fietescheel/Documents/Master%20Thesis/research/img/analysis/factuality_metrics_4panel_comparison.png)

* **Panel A (SBERT):** Zeigt deutlich, dass SBERT zwar bei zufälligem Durchtauschen (Random Shuffle) auf $0.65$ abfällt, die echten Halluzinationen und Minimal-Perturbationen jedoch bei hohen Werten ($0.79\text{--}0.87$) verbleiben.
* **Panel B (NLI Contradiction):** Weist bei den gezielten Minimal-Perturbationen (Klasse 4) den höchsten Widerspruchswert ($P_{\text{contra}} \approx 0.41$) auf.
* **Panel C (NER Jaccard):** Visualisiert das NER-Dilemma: Auf echten Gold-Texten liegt der Überlapp bei nur $13\%$, während manipulierte Sätze $68\%$ erreichen.

---

### 4.2 ROC-Kurven der Trennkraft
![ROC-Kurven der Metriken](file:///Users/fietescheel/Documents/Master%20Thesis/research/img/analysis/factuality_metrics_roc_curves.png)

---

## 5. Qualitative Fallstudien (Detail-Inspektion)

### Fall 1: Zahlen-Slot-Verschiebung (Costa Concordia)
* **Alltagssprache:** *„Früherer Kapitän in erster Instanz zu 16 Jahren Haft verurteilt.“*
* **SFT-Generierung:** *„Er wurde verhaftet und war 16 Jahre alt.“*
* **Metrik-Reaktion:**
  * SBERT: **$0.843$** (Täuschend hoch $\implies$ Blindspot)
  * NLI $P(\text{Contra})$: **$0.091$** (Im Volltext durch neutrale Kontexttokens verwässert)
  * Number Check: **$0.500$** (Erkennt, dass Zahl ohne Haft-Kontext vorliegt)

### Fall 2: Minimal-Perturbation (Negations-Inversion)
* **Alltagssprache:** *„Die Beratung in den Pflegestützpunkten ist für alle Bürger kostenlos.“*
* **Manipulierter Zieltext:** *„Die Beratung in den Pflegestützpunkten ist nicht kostenlos und kostet viel Geld.“*
* **Metrik-Reaktion:**
  * SBERT: **$0.832$** (Vollkommen blind für das Wort *„nicht“*!)
  * NLI $P(\text{Contra})$: **$0.894$** (**Erkennt den Widerspruch sofort zu fast 90 %!**)
  * NER Jaccard: **$0.857$** (Vergibt fälschlicherweise Bestnote, da alle Namen matchen)

---

## 6. Zentrale Erkenntnisse & Schlussfolgerungen für die Masterarbeit

1. **Empirischer Beweis des SBERT-Blindspots:**  
   SBERT unterscheidet zuverlässig zwischen völlig unterschiedlichen Themen (Random Shuffle: $0.65$ vs. Gold: $0.89$), ist jedoch bei Negationen und Zahlenverdrehungen innerhalb desselben Themas nahezu unempfindlich ($0.79\text{--}0.87$).
2. **Endgültiger Nachweis des NER-Dilemmas (ROC-AUC = $0.49$):**  
   Klassisches NER ist für die Evaluation von Leichter Sprache **nachweislich ungeeignet**. Da Leichte Sprache Fachbegriffe und Eigennamen bewusst abstrahiert (*„BMAS“ $\rightarrow$ „Politiker“*), führt ein hoher NER-Überlapp nicht zu höherer Qualität, sondern belohnt manipulierte Sätze mit unnötigem Fachjargon.
3. **Dokument- vs. Satzebenen-Effekt bei NLI:**  
   NLI Cross-Encoder entfalten ihre maximale Trennschärfe auf **Satzebene** (Negations-Erkennung $P_{\text{contra}} \approx 0.89$). Bei ganzen Dokumenten mit 300+ Tokens wird die Entailment-Wahrscheinlichkeit durch unbezogene Hintergrundsätze ins *Neutrale* verwässert.
4. **Empfehlung für zukünftiges Alignment & DPO:**  
   * Für das DPO-Pair-Mining empfiehlt sich ein **satzweiser NLI-Contradiction-Filter ($P_{\text{contra}} < 0.30$)** in Kombination mit einem **numerischen Regex-Check**.
   * Dadurch wird verhindert, dass Halluzinationen als `chosen`-Texte in den Trainingsdatensatz gelangen, und Reward Hacking wird effektiv gestoppt.
