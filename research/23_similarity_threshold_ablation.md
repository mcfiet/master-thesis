# Experiment 26: Semantische Ähnlichkeits-Schwellenwerte & Sweet-Spot-Ablation (0.60 vs. 0.70 vs. 0.80 bis 0.98)

Dieses Dokument dokumentiert die theoretischen Grundlagen, das experimentelle Setup, die empirischen Fragestellungen und die wissenschaftliche Analyse des Experiments zur **Ablation der semantischen Ähnlichkeits-Schwellenwerte** für den **MixUp-Regressor** und das **SFT-Übersetzungsmodell** (`facebook/mbart-large-50`).

---

## 1. Theoretischer Hintergrund & Forschungsfrage (FF 2.2)

Beim Aufbau paralleler Textkorpora für die deutsche Leichte Sprache aus heterogenen Webquellen stehen Forscher vor einem fundamentalen Zielkonflikt:

$$\text{Trade-off: Maximale Datenmenge (Sample Volume) vs. Hohe Datenreinheit (Semantic Purity)}$$

* **Gefahr zu niedriger Schwellenwerte ($s_{\min} < 0{,}70$):**  
  Toleriert man Paare mit geringer semantischer Ähnlichkeit ($0{,}60 \le \text{sim} < 0{,}70$), erhält man zwar das maximale Datenvolumen ($N = 867$ Paare), riskiert jedoch die Kontamination des Trainingsbestands durch thematische Fehl-Alignments, radikale Teaser-Kürzungen oder CMS-Fehlverlinkungen.
* **Gefahr zu hoher Schwellenwerte ($s_{\min} \ge 0{,}85$ / $0{,}90$):**  
  Filtert man zu strikt, sinkt die Sample-Anzahl dramatisch ab ($N = 144$ Paare bei $s_{\min} = 0{,}90$, ein Verlust von über $83\,\%$). Zudem weisen Textpaare mit extrem hoher Ähnlichkeit ($\text{sim} > 0{,}90$) häufig kaum Vereinfachungsmerkmale auf, wodurch generative Modelle eine triviale Identitätsfunktion lernen (*Data Starvation & Copy Bias*).

### Die zentrale Forschungsfrage (FF 2.2):
> *Welcher minimale Filterbereich der Kosinus-Ähnlichkeit ($0{,}60$, $0{,}70$ oder $0{,}80$ bis $0{,}98$) liefert die qualitativ und quantitativ beste Grundlage für das Training (a) der kontinuierlichen Komplexitätsmetrik (MixUp-Regressor) und (b) des generativen Übersetzungsmodells (SFT)?*

---

## 2. Experimentelles Ceteris-Paribus-Setup

Alle Experimente nutzen die **exakten Standard-Hyperparameter aus der Hauptpipeline (`run_pipeline`)**:

### A. Metrik-Modell: BiLSTM MixUp Regressor
* **Architektur:** `BiLSTMRegressor` (Embedding: 128, Hidden: 128, Dropout: 0.3, Sigmoid)
* **Sequenzlänge:** `max_seq_len = 1024`
* **MixUp Multiplikator:** `mixtures_per_pair = 160`
* **Dataloader:** Hybrid (Curriculum-Shift von statischen zu dynamischen Mischungen)
* **Batch Size & Epochen:** `batch_size = 64`, `epochs = 80`, `lr = 0.001`, `patience = 15`
* **Scheduler:** `CosineAnnealingWarmRestarts` ($T_0 = 10, T_{\text{mult}} = 1, \eta_{\min} = 10^{-5}$)

### B. Generatives Modell: SFT mBART-50 LoRA
* **Basismodell:** `facebook/mbart-large-50` (mit de_DE Sprachanker)
* **PEFT / LoRA:** $r = 16, \alpha = 32, \text{dropout} = 0.05$ (Attention + FC Projektionen)
* **Sequenzlänge:** `max_source_len = 1024`, `max_target_len = 1024`
* **Batch Size & Optimierung:** `batch_size = 2`, `accumulation_steps = 8` (Effektiv: 16), `lr = 1e-4`, Linear Warmup (10%), `epochs = 30`, `patience = 10`
* **Inferenz Setup:** Beam Search (`num_beams = 4`), `repetition_penalty = 1.2`, `no_repeat_ngram_size = 3`

---

## 3. Untersuchte Schwellenwert-Bänder im Master-Korpus

| Versuchsband | Schwellenwert $s_{\min} \dots s_{\max}$ | Artikelpaare ($N$) | Retention (%) | AS Tokens | LS Tokens | Token-Ratio (LS/AS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Set A (0.60)** | $[0{,}60, 0{,}98]$ | 867 | 99,9 % | 523.097 | 385.624 | 0,737 |
| **Set B (0.70)** | $[0{,}70, 0{,}98]$ | 803 | 92,5 % | 484.631 | 362.899 | 0,749 |
| **Set C (0.80)** | $[0{,}80, 0{,}98]$ | 643 | 74,1 % | 403.356 | 310.487 | 0,770 |
| **Set D (0.85)** | $[0{,}85, 0{,}98]$ | 431 | 49,7 % | 275.761 | 214.993 | 0,780 |
| **Set E (0.90)** | $[0{,}90, 0{,}98]$ | 144 | 16,6 % | 83.462 | 79.905 | 0,957 |

---

## 4. Evaluierungs-Methodik & Metriken

Die Evaluation erfolgt zweistufig für beide Modelltypen:

1. **In-Domain Evaluation (Fixierter 10%-Held-Out-Testsplit):**
   * *MixUp Regressor:* Mean Squared Error ($MSE$), Mean Absolute Error ($MAE$), Bestimmtheitsmaß ($R^2$), Pearson-Korrelation ($r$), Spearman-Rangkorrelation ($\rho$).
   * *SFT Modell:* Bester Validierungs-Loss, Finaler Trainings-Loss.

2. **Out-of-Domain (OOD) Evaluation auf realem *Lebenshilfe*-Benchmark (`lebenshilfe_dataset_clean.json`):**
   * *MixUp Regressor:*
     - Separation ROC-AUC zwischen echten AS-Dokumenten ($\lambda=0.0$) und echten LS-Dokumenten ($\lambda=1.0$).
     - Out-of-Domain $MAE$ und $MSE$.
     - Perfect Pair Match Rate (Anteil der parallelen Dokumentenpaare, bei denen $\text{Score}(LS) > \text{Score}(AS)$ gilt).
     - Mittlere Vorhersagewerte $\bar{\lambda}_{AS}$ und $\bar{\lambda}_{LS}$ sowie Spreizungs-Delta $\Delta = \bar{\lambda}_{LS} - \bar{\lambda}_{AS}$.
   * *SFT Modell:*
     - Stilistische Einfachheit $R_{\text{style}}$ (vorhergesagt durch den BiLSTM-MixUp-Regressor).
     - Semantischer Erhalt zur Ausgangssprache $R_{\text{sem, AS}}$ (Long-Context Jina-SBERT).
     - Ähnlichkeit zur menschlichen Leichte-Sprache-Referenz $\text{Sim}_{\text{ref}}$.
     - Gesamt-Reward $\text{Composite Reward} = 0{,}5 \cdot R_{\text{style}} + 0{,}5 \cdot R_{\text{sem}}$.
     - Lexikalische Überlappung: BLEU und ROUGE-L.
     - Qualitätsindikatoren: Truncation Rate (%) und durchschnittliche Token-Länge.

---

## 5. Wissenschaftliche Hypothesen & Erwartete Erkenntnisse

### Hypothese 1: Divergenz der optimalen Schwellenwerte zwischen Regressor und Übersetzer
* **MixUp Regressor profitiert von $s_{\min} = 0{,}80$:**  
  Da der Regressor aus synthetischen Satzmischungen lineare Sprachkomplexitäts-Gradienten ($\lambda$) berechnet, reagiert er extrem sensibel auf inhaltliche Asymmetrien. Sind in einem Paar mit $s=0{,}62$ Absätze unvollständig übersetzt oder thematisch verschoben, lernt der Regressor fehlerhafte Labels. Die striktere Filterung ($0{,}80$) liefert die höchste OOD-Trennschärfe.
* **SFT Übersetzungsmodell profitiert von $s_{\min} = 0{,}70$:**  
  Autoregressive Sequence-to-Sequence Modelle mit vortrainiertem Sprachverständnis (mBART-50) sind robuster gegenüber leichten semantischen Varianzen und profitieren maßgeblich von einer breiteren lexikalischen Diversität und größeren Textmenge ($803$ vs. $643$ Artikelpaare). Erst unterhalb von $0{,}70$ überwiegen Alignment-Fehler und führen zu Halluzinationen.

### Hypothese 2: Zusammenbruch bei extremer Filterung ($s_{\min} \ge 0{,}85$)
* Ein Schwellenwert von $s_{\min} \ge 0{,}85$ oder $0{,}90$ führt zu einem massiven Sample-Verlust ($>50\,\%$ bis $83\,\%$). Die verbleibenden Texte sind stilistisch oft zu nah am Original, sodass das Modell verlernt, komplexe Sätze aufzubrechen.

---

## 6. Verknüpfte Skripte & Artefakte

* **MixUp Trainingsskript:** [`scripts/experiments/similarity_threshold/train_similarity_mixup.py`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/experiments/similarity_threshold/train_similarity_mixup.py)
* **SFT Trainingsskript:** [`scripts/experiments/similarity_threshold/train_similarity_sft.py`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/experiments/similarity_threshold/train_similarity_sft.py)
* **Konsolidierungs- & Evaluationsskript:** [`scripts/experiments/similarity_threshold/evaluate_all_similarity_thresholds.py`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/experiments/similarity_threshold/evaluate_all_similarity_thresholds.py)
* **SLURM Batch-Runner:** [`scripts/sbatch/experiments/similarity_threshold/run_all_similarity_experiments.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/experiments/similarity_threshold/run_all_similarity_experiments.sh)
* **Analyse-Notebook:** [`notebooks/experiments/similarity_threshold.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/experiments/similarity_threshold.ipynb)
* **Ergebnisverzeichnis:** `results/experiments/similarity_threshold/`
