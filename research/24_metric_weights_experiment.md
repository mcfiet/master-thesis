# Experiment 24: Untersuchung der Reward-Gewichtung (Simplicity vs. Semantik) für das Encoder-Decoder-Modell

Dieses Experiment untersucht den systematischen Einfluss der Gewichtung in der Verbund-Reward-Funktion (*Composite Reward*) auf das *Direct Preference Optimization* (DPO) Training des Encoder-Decoder-Übersetzungsmodells (`facebook/mbart-large-50`).

---

## 1. Motivation & Forschungsfrage

In der automatischen Textvereinfachung von Alltagssprache (AS) in Leichte Sprache (LS) stehen zwei Zielgrößen in einem inhärenten Spannungsverhältnis:

1. **Einfachheit & Stil ($R_{\text{style}}$):** Kurze Sätze, einfacher Satzbau (SVO), leicht verständliches Vokabular, Vermeidung von Nebensätzen und Passivkonstruktionen.
2. **Semantischer Erhalt & Faktentreue ($R_{\text{sem}}$):** Vollständige Beibehaltung der Kernbedeutung und Fakten der Ausgangssprache (AS) ohne unerwünschte Auslassungen oder Halluzinationen.

### Forschungsfragen:
* **F1:** Wie verändern sich die Translationsqualität und der Lesbarkeitsgrad, wenn die Simplicity-Gewichtung von $w_{\text{style}} = 0.5$ auf $w_{\text{style}} = 0.7$ bzw. $w_{\text{style}} = 1.0$ erhöht wird?
* **F2:** Ab welchem Gewichtungspunkt schlägt der Fokus auf Einfachheit in aggressiven Informationsverlust oder Satzabbrüche (*Truncation*) um?
* **F3:** Welches Gewichtungsverhältnis bildet die optimale Pareto-Grenze zwischen syntaktischer Einfachheit und semantischer Konsistenz für Encoder-Decoder-Architekturen?

---

## 2. Mathematische Formulierung der Verbund-Reward-Funktion

Für jedes generierte Übersetzungskandidaten-Paar $(y_w, y_l)$ bei gegebenem Quelltext $x$ wird der Gesamtwert nach folgender Formel bestimmt:

$$R(x, y) = w_{\text{style}} \cdot R_{\text{style}}(y) + w_{\text{sem}} \cdot R_{\text{sem, norm}}(x, y)$$

wobei:
* $R_{\text{style}}(y) \in [0, 1]$ die Vorhersage des trainierten BiLSTM-MixUp-Regressors darstellt.
* $R_{\text{sem, norm}}(x, y) = \frac{\cos(\mathbf{e}_x, \mathbf{e}_y) + 1}{2} \in [0, 1]$ die normalisierte Kosinus-Ähnlichkeit der Sentence-BERT-Embeddings (`paraphrase-multilingual-mpnet-base-v2`) ist.
* $w_{\text{style}} + w_{\text{sem}} = 1.0$.

### Getestete Konfigurationen:

| Konfiguration | $w_{\text{style}}$ (Simplicity) | $w_{\text{sem}}$ (Semantik) | Fokus / Hypothese |
| :--- | :---: | :---: | :--- |
| **`w05_w05` (Standard)** | $0.50$ | $0.50$ | Ausgewogene Balance zwischen Stil und Information. |
| **`w07_w03` (Simplicity-Priorität)** | $0.70$ | $0.30$ | Stärkere Vereinfachung bei akzeptabler semantischer Führung. |
| **`w10_w00` (Reine Simplicity)** | $1.00$ | $0.00$ | Maximale Reduktion auf Leichte-Sprache-Muster ohne explizite Semantik-Strafe. |

---

## 3. Experimenteller Aufbau & Pipeline

Die Pipeline ist modular aufgebaut und führt folgende Schritte automatisiert aus:

```
                               ┌──────────────────────────────────────────────────────────┐
                               │  SFT Baseline Modell (facebook/mbart-large-50 + LoRA)   │
                               └────────────────────────────┬─────────────────────────────┘
                                                            │
                     ┌──────────────────────────────────────┼──────────────────────────────────────┐
                     ▼                                      ▼                                      ▼
      [Track 1: w_style=0.5, w_sem=0.5]      [Track 2: w_style=0.7, w_sem=0.3]      [Track 3: w_style=1.0, w_sem=0.0]
      ├── 1_generate_dpo_pairs_w05_w05       ├── 1_generate_dpo_pairs_w07_w03       ├── 1_generate_dpo_pairs_w10_w00
      │   (data/.../dpo_pairs_w05_w05)       │   (data/.../dpo_pairs_w07_w03)       │   (data/.../dpo_pairs_w10_w00)
      │                                      │                                      │
      ▼                                      ▼                                      ▼
      ├── 2_train_dpo_w05_w05                ├── 2_train_dpo_w07_w03                ├── 2_train_dpo_w10_w00
      │   (results/.../dpo_w05_w05)          │   (results/.../dpo_w07_w03)          │   (results/.../dpo_w10_w00)
      └──────────────────────────────────────┼──────────────────────────────────────┘
                                             │
                                             ▼
                     ┌──────────────────────────────────────────────────────────┐
                     │ 3_run_full_evaluation.sh                                 │
                     │ - Benchmark auf Lebenshilfe-Testset                      │
                     │ - R_style, R_sem_AS, Sim_ref, BLEU, ROUGE-L, Länge       │
                     │ - Trade-off Plots & Pareto-Kurven                        │
                     └──────────────────────────────────────────────────────────┘
```

---

## 4. Auszuführende Skripte & SLURM-Jobs

Alle Skripte befinden sich unter [`scripts/sbatch/experiments/metric_weights/`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/experiments/metric_weights).

### Automatische Ausführung aller Tracks:
```bash
bash scripts/sbatch/experiments/metric_weights/run_all_metric_weights_experiments.sh
```

### Manuelle Einzelschritte:

1. **Präferenzpaare generieren:**
   ```bash
   sbatch scripts/sbatch/experiments/metric_weights/1_generate_dpo_pairs_w05_w05.sh
   sbatch scripts/sbatch/experiments/metric_weights/1_generate_dpo_pairs_w07_w03.sh
   sbatch scripts/sbatch/experiments/metric_weights/1_generate_dpo_pairs_w10_w00.sh
   ```

2. **DPO Modelle trainieren:**
   ```bash
   sbatch scripts/sbatch/experiments/metric_weights/2_train_dpo_w05_w05.sh
   sbatch scripts/sbatch/experiments/metric_weights/2_train_dpo_w07_w03.sh
   sbatch scripts/sbatch/experiments/metric_weights/2_train_dpo_w10_w00.sh
   ```

3. **Gesamtevaluation durchführen:**
   ```bash
   sbatch scripts/sbatch/experiments/metric_weights/3_run_full_evaluation.sh
   ```

---

## 5. Zielmetriken & Evaluationsergebnisse

Die Ergebnisse werden automatisch in folgenden Artefakten gespeichert:
* Zusammenfassung: [`results/evaluation/metric_weights_comparison_summary.csv`](file:///Users/fietescheel/Documents/Master%20Thesis/results/evaluation/metric_weights_comparison_summary.csv)
* Detaillierte Satzübersetzungen: [`results/evaluation/metric_weights_comparison_details.csv`](file:///Users/fietescheel/Documents/Master%20Thesis/results/evaluation/metric_weights_comparison_details.csv)
* Grafische Darstellung: [`results/plots/metric_weights_tradeoff_curve.png`](file:///Users/fietescheel/Documents/Master%20Thesis/results/plots/metric_weights_tradeoff_curve.png)
