# Experiment 25: DPO-Loss-Aggregation – Sum vs. Mean (Length-Normalized DPO)

## 1. Theoretischer Hintergrund & Problemstellung

Im Direct Preference Optimization (DPO) Framework wird die Präferenz zwischen einem präferierten Text $y_w$ (*winner*) und einem dispräferierten Text $y_l$ (*loser*) anhand der relativen Log-Likelihoods gegenüber einem Referenzmodell $\pi_{\text{ref}}$ bewertet:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right) \right]$$

### Das Problem der klassischen Summierung (`sum`):
Standardmäßig berechnet DPO die Sequenz-Wahrscheinlichkeit als Summe der Token-Log-Wahrscheinlichkeiten:
$$\log \pi(y \mid x) = \sum_{t=1}^{|y|} \log \pi(y_t \mid x, y_{<t})$$

Da Wahrscheinlichkeiten $p \in (0, 1]$ gelten, sind Log-Wahrscheinlichkeiten immer **negativ** ($\log p \le 0$).
* Eine lange, wohlformulierte Übersetzung in Leichter Sprache ($|y| = 150$ Tokens) akkumuliert 150 negative Terme (z. B. $150 \times (-0.15) = -22.5$).
* Ein kurzer, trivialer Satzabbruch ($|y| = 25$ Tokens) akkumuliert nur 25 negative Terme (z. B. $25 \times (-0.15) = -3.75$).

**Folge:** Das Modell lernt eine Kürzungsfalle (*Length Exploitation* / Degeneration): Es beendet die Generierung nach 25–30 Tokens sofort mit dem `</s>`-Token, um der mathematischen Akkumulation negativer Log-Probabilities auszuweichen.

---

### Die Lösung: Längen-Normalisierung (`mean` / Per-Token DPO):
Durch Normalisierung mit der tatsächlichen Sequenzlänge $|y|$ wird der reine Längen-Einfluss eliminiert:
$$\overline{\log \pi(y \mid x)} = \frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi(y_t \mid x, y_{<t})$$

Bewertet wird damit die **durchschnittliche Modellzuversicht pro generiertem Token**. Ein langer Text wird nicht mehr dafür bestraft, dass er lang ist.

---

## 2. Experimentelles Setup (Ceteris Paribus)

| Parameter | Konfiguration |
| :--- | :--- |
| **Basismodell** | `facebook/mbart-large-50` (SFT Baseline aus `results/models/sft`) |
| **PEFT / LoRA** | $r=16, \alpha=32$, Dropout $0.05$, Target Modules: Attention + FC |
| **Lernrate & Scheduler** | $5 \cdot 10^{-6}$, AdamW, Linear Warmup (10%) |
| **DPO-Beta ($\beta$)** | $0.1$ |
| **Epochen / Batch Size** | 3 Epochen, Batch Size 2, Gradient Accumulation 8 |
| **Datensatz** | Lebenshilfe DPO-Präferenzpaare (`dpo_pairs_w05_w05.jsonl`) |

### Untersuchte Modellvarianten:
1. **SFT Baseline:** Referenzmodell ohne DPO (`results/models/sft`)
2. **DPO Sum (Classic):** `--loss_type sum` (`results/models/loss_aggregation_exp/dpo_sum`)
3. **DPO Mean (Length-Normalized):** `--loss_type mean` (`results/models/loss_aggregation_exp/dpo_mean`)

---

## 3. Ausführung auf dem Compute-Cluster

### Alle Trainings- und Evaluationsjobs mit Abhängigkeiten starten:
```bash
bash scripts/sbatch/experiments/loss_aggregation/run_all_loss_aggregation_experiments.sh
```

### Einzelne Jobs starten:
```bash
# 1. DPO Sum
sbatch scripts/sbatch/experiments/loss_aggregation/1_train_dpo_sum.sh

# 2. DPO Mean
sbatch scripts/sbatch/experiments/loss_aggregation/1_train_dpo_mean.sh

# 3. Evaluation
sbatch scripts/sbatch/experiments/loss_aggregation/2_run_full_evaluation.sh
```

---

## 4. Evaluierung & Analyse

Das Notebook **[`notebooks/research/translation/analyse_loss_aggregation_experiment.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/research/translation/analyse_loss_aggregation_experiment.ipynb)** vergleicht:
1. **Ø Generierte Textlänge:** Bleibt die Textlänge bei `mean` stabil erhalten (vs. Einbruch bei `sum`)?
2. **Satzabbruch-Quote (Truncation Rate %):** Sinkt die Quote unvollständiger Sätze?
3. **Simplicity-Score ($R_{\text{style}}$):** Erreicht `mean` höhere Werte durch vollständige Leichte-Sprache-Sätze?
4. **Semantischer Erhalt ($R_{\text{sem, AS}}$) & BLEU / ROUGE-L.**
