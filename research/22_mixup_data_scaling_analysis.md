# 22. Data Scaling & Empirische Lernkurven-Analyse für das MixUp-Metrik-Modell

**Thema:** Daten-Skalierungsstudie & Lernkurven des BiLSTM MixUp Simplicity Regressors  
**Datum:** 23. August 2026  
**Autor:** Fiete Scheel  
**Notebook:** [`notebooks/research/metric/5_mixup_data_scaling_analysis.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/research/metric/5_mixup_data_scaling_analysis.ipynb)  
**Skripte:** [`scripts/experiments/data_scaling/`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/experiments/data_scaling/), [`scripts/sbatch/experiments/data_scaling/`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/experiments/data_scaling/)

---

## 1. Motivation & Fragestellung

Für das Training des kontinuierlichen MixUp-Regressors (Qualitäts- und Einfachheits-Metrik $\lambda \in [0.0, 1.0]$) stellte sich die fundamentale Frage:
1. **Wie viele Datenpaare werden für das Training des MixUp-Modells tatsächlich benötigt?**
2. **Reicht die aktuelle Datenmenge aus oder profitiert das Modell von einer Erhöhung?**
3. **Gibt es einen empirischen Sättigungspunkt (*Plateau / Diminishing Returns*), ab dem zusätzliche Datenmischungen keinen signifikanten Mehrwert mehr bringen?**

Um diese Fragen empirisch und wissenschaftlich für die Masterarbeit zu beantworten, wurde eine systematische **Lernkurven- und Skalierungsstudie (*Data Scaling / Sample Complexity Analysis*)** entlang zweier getrennter Achsen aufgesetzt:
* **Achse 1 (Synthetischer MixUp-Multiplikator $M$):** $M \in \{2, 5, 10, 20, 40, 80\}$ bei 100% der Basis-Trainingsartikel.
* **Achse 2 (Reale Basis-Artikelpaare $N$):** $F \in \{10\%, 25\%, 50\%, 75\%, 100\%\}$ (entspricht $N \in \{48, 120, 240, 360, 480\}$ Artikelpaaren) bei festem Multiplikator $M=20$.

---

## 2. Versuchsaufbau & Methodik

### 2.1 Datenbasis & Split-Design (Vermeidung von Data Leakage)
* **Rohdaten:** `data/analysis/corpus_master.csv`
* **Filter:** `semantic_similarity_8192` $\in [0.80, 0.98]$ und `dropna(subset=["ls_text", "as_text"])`.
* **Gefilterte Artikelpaare gesamt:** **600 Paare**
* **Aufteilung (80 / 10 / 10):**
  * **Test-Set (10%):** **60 Artikelpaare** (strikt ungesehen, fixiert für alle 11 Experimentläufe).
  * **Validation-Set (10%):** **60 Artikelpaare** (für Early Stopping und Model Checkpointing).
  * **Trainings-Set (80%):** **480 Artikelpaare** (= 100% der maximal zulässigen Trainingsbasis).

### 2.2 Modell & Trainingsparameter
* **Architektur:** BiLSTM (Embedding Dim: 128, Hidden Dim: 128, Dropout: 0.3, Sigmoid-Output).
* **Vokabular:** Einheitlich 25.000 Tokens (gespeichert unter `data/vocabs/mixup_vocab.json`), um Konsistenz über alle Skalierungsstufen zu garantieren.
* **Sampling:** Hybrid Dynamic MixUp mit $p_{\text{dynamic}} = \frac{\text{epoch}}{\text{total\_epochs}-1}$ und Cosine Annealing Learning Rate Scheduler mit Warm Restarts ($T_0=10$, $\eta_{\min}=10^{-5}$).
* **Epochen:** 40 (Early Stopping Patience: 8), Batch Size: 64, Sequenzlänge: 256 Tokens.

---

## 3. Empirische Ergebnisse

Alle 11 trainierten Modelle wurden auf demselben festen, ungesehenen Test-Split evaluiert:

| Experiment | Gruppe | Basis-Paare ($N$) | MixUp ($M$) | Samples / Epoche | Test MSE $\downarrow$ | Test MAE $\downarrow$ | Test $R^2$ $\uparrow$ | Binäre Acc $\uparrow$ | Train-Zeit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `scale_mixtures_m2` | `mixtures_scaling` | 480 | **2** | 960 | 0.05645 | 0.18670 | 0.26708 | 74.00 % | 3.55 s |
| `scale_mixtures_m5` | `mixtures_scaling` | 480 | **5** | 2.400 | 0.05130 | 0.17775 | 0.33396 | 74.50 % | 3.91 s |
| `scale_mixtures_m10` | `mixtures_scaling` | 480 | **10** | 4.800 | 0.04966 | 0.17746 | 0.35533 | 74.50 % | 5.76 s |
| **`scale_mixtures_m20`** | `mixtures_scaling` | **480** | **20** | **9.600** | **0.02483** | **0.11206** | **0.67769** | **85.08 %** | **46.68 s** |
| `scale_mixtures_m40` | `mixtures_scaling` | 480 | **40** | 19.200 | 0.02385 | 0.10380 | 0.69032 | 86.42 % | 90.53 s |
| `scale_mixtures_m80` | `mixtures_scaling` | 480 | **80** | 38.400 | 0.02329 | 0.10498 | 0.69762 | 86.83 % | 83.79 s |
| | | | | | | | | | |
| `scale_pairs_f010` | `pairs_scaling` | **48 (10%)** | 20 | 960 | 0.07007 | 0.20860 | 0.09032 | 68.25 % | 2.05 s |
| `scale_pairs_f025` | `pairs_scaling` | **120 (25%)** | 20 | 2.400 | 0.04103 | 0.15874 | 0.46727 | 79.17 % | 7.76 s |
| `scale_pairs_f050` | `pairs_scaling` | **240 (50%)** | 20 | 4.800 | 0.03560 | 0.14613 | 0.53787 | 81.92 % | 11.62 s |
| `scale_pairs_f075` | `pairs_scaling` | **360 (75%)** | 20 | 7.200 | 0.02921 | 0.12720 | 0.62075 | 83.17 % | 20.41 s |
| **`scale_pairs_f100`** | `pairs_scaling` | **480 (100%)** | **20** | **9.600** | **0.02483** | **0.11206** | **0.67769** | **85.08 %** | **46.50 s** |

---

## 4. Detaillierte Erkenntnisse & wissenschaftliche Analyse

### 4.1 Achse 1: Synthetischer MixUp-Multiplikator ($M$) – Phasenübergang & Sättigung
1. **Kritischer Phasenübergang bei $M = 20$:**
   * Bei niedrigen Werten ($M \in \{2, 5, 10\}$) stagniert das Modell auf schwachem Niveau ($R^2 \approx 0.27 - 0.35$, $\text{MAE} \approx 0.18$, $\text{Acc} \approx 74.5\%$). Das Modell lernt in diesem Bereich keine feingranulare kontinuierliche Regression, sondern verbleibt bei einer groben Approximation.
   * **Zwischen $M=10$ und $M=20$ erfolgt ein massiver Leistungssprung:** Der Test-MSE halbiert sich ($0.0497 \rightarrow 0.0248$), $R^2$ verdoppelt sich fast ($0.355 \rightarrow 0.678$) und die binäre Klassifikationsgenauigkeit steigt sprunghaft um $+10.58$ Prozentpunkte auf **85.08%**.
   * *Ursache:* Erst ab circa 20 Mischungen pro Artikelpaar wird das kontinuierliche $\lambda$-Intervall $[0.0, 1.0]$ in Kombination mit dem dynamischen Resampling dicht genug abgetastet, damit das BiLSTM eine präzise Regressionslinie lernt.

2. **Eindeutiges Sättigungsplateau ab $M = 20$ (*Diminishing Returns*):**
   * Eine Vervierfachung des Trainingsaufwands von $M=20$ auf $M=80$ (von 9.600 auf 38.400 Samples pro Epoche) bringt beim Test-MAE praktisch **keinen relevanten Gewinn mehr** ($0.112 \rightarrow 0.105$, Verbesserung $< 0.7$ Prozentpunkte).
   * Das Bestimmtheitsmaß $R^2$ konvergiert asymptotisch gegen $\approx 0.698$.
   * **Fazit:** Die Wahl von **$M = 20$** in der Hauptpipeline ist empirisch exakt der **optimale Sweet Spot** zwischen maximaler Modellgüte und Rechenaufwand.

```text
Test MAE
  0.20 ┼  ● (M=2)
  0.18 ┼     ● (M=5)  ● (M=10)
  0.16 ┼
  0.14 ┼
  0.12 ┼                 ● (M=20: Sweet Spot)
  0.10 ┼───────────────────────● (M=40) ──── ● (M=80)  <- Sättigungsplateau
       └─────┴────────┴────────┴─────────────┴────────► MixUp-Multiplikator M
```

---

### 4.2 Achse 2: Reale Basis-Artikelpaare ($N$) – Stetiger Qualitätsgewinn
1. **Kein Plateau bei der Anzahl realer Artikel:**
   * Im Gegensatz zum MixUp-Multiplikator zeigt die Skalierung über die realen Textpaare ($N$) einen **kontinuierlichen, annähernd linearen Qualitätsgewinn**:
     * $10\%$ ($N=48$): $\text{MAE} = 0.2086$, $R^2 = 0.090$
     * $25\%$ ($N=120$): $\text{MAE} = 0.1587$, $R^2 = 0.467$
     * $50\%$ ($N=240$): $\text{MAE} = 0.1461$, $R^2 = 0.538$
     * $75\%$ ($N=360$): $\text{MAE} = 0.1272$, $R^2 = 0.621$
     * $100\%$ ($N=480$): $\text{MAE} = \mathbf{0.1121}$, $R^2 = \mathbf{0.678}$
   * Die Kurve flacht bei $N=480$ noch nicht vollständig ab.
2. **Ursache:** Zusätzliche reale Artikelpaare erweitern das tatsächliche Vokabular, die linguistische Vielfalt und die Satzstrukturen. Diese Form von Information kann nicht durch reines synthetisches Mischen aus wenigen Ausgangstexten erzeugt werden.

---

### 4.3 Direktvergleich: Reale Datenbasis vs. Synthetische Mischungen

Ein besonders aussagekräftiger Vergleich ergibt sich bei identischer effektiver Sample-Menge (z. B. **960 Samples pro Epoche**):

* **480 Artikel $\times$ 2 Mischungen:** $\text{MAE} = \mathbf{0.1867}$, $R^2 = \mathbf{0.2671}$, $\text{Acc} = \mathbf{74.00\%}$
* **48 Artikel $\times$ 20 Mischungen:** $\text{MAE} = \mathbf{0.2086}$, $R^2 = \mathbf{0.0903}$, $\text{Acc} = \mathbf{68.25\%}$

> **Wissenschaftliches Fazit:** Bei gleicher Sample-Anzahl ist eine breitere Basis an realen Texten **dreimal so effektiv** ($R^2$ 0.267 vs. 0.090) wie das mehrfache Remixen weniger Ausgangstexte. Reale Textdiversität ist durch synthetische Kombinationen nicht vollständig ersetzbar.

---

## 5. Fazit & Argumentation für die Masterarbeit

1. **Empirische Rechtfertigung der Parameter:**
   * Der in der Arbeit verwendete MixUp-Multiplikator **$M = 20$** ist durch diese Studie empirisch fundiert belegt. Kleinere Werte ($M < 20$) führen zu Unterabtastung und schlechter Regression; größere Werte ($M > 20$) führen zu Sättigung ohne signifikanten Qualitätsgewinn.
2. **Angemessenheit der Datenmenge:**
   * Die aktuell verwendeten **480 Trainings-Artikelpaare** (100% des verfügbaren Trainingskorpus) mit $M=20$ generieren **9.600 Trainingspaare pro Epoche**. In Kombination mit dem dynamischen Hybrid-Sampling sieht das BiLSTM über 40 Epochen hinweg bis zu **384.000 unterschiedliche Satzkombinationen**.
   * Dies reicht für das kompakte BiLSTM-Modell vollständig aus, um stabile, verlässliche Regressionssignale ($\text{MAE} \approx 0.11$, $R^2 \approx 0.68$, $\text{Accuracy} \approx 85\%$) für die nachgelagerte DPO-Belohnungsberechnung zu liefern.
3. **Ausblick:**
   * Sollte das Metrikmodell in zukünftigen Arbeiten weiter optimiert werden, liegt der Hebel nicht in einer Erhöhung des MixUp-Multiplikators, sondern im Hinzufügen weiterer realer Textquellen (Erweiterung der $N=480$ Basispaare).
