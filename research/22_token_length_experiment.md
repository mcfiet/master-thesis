# 22. Versuchsaufbau & Evaluation: Einfluss der Token-Länge (256 vs. 500 vs. 1000 Tokens)

## 1. Motivation & Forschungsfrage

In bisherigen Experimenten wurde standardmäßig mit Sequenzlängen von $256$ Tokens gearbeitet. Deutsche Artikel in Alltagssprache (AS) sowie deren Entsprechungen in Leichter Sprache (LS) umfassen im Lebenshilfe-Korpus jedoch durchschnittlich $\approx 916$ bzw. $\approx 1107$ Tokens.

Daraus ergibt sich die zentrale Forschungsfrage:
> **Welchen Einfluss hat die maximale Token-Länge ($256$, $500$, $1000$ Tokens) auf die Modellierungsqualität der drei Pipeline-Stufen:**
> 1. **Simplicity-Metrik (BiLSTM MixUp Regressor)**
> 2. **SFT Fine-Tuning (mBART-50 LoRA)**
> 3. **DPO Alignment (Direct Preference Optimization)**

---

## 2. Experimenteller Aufbau & Methodik

### 2.1 Hardware- & Ressourcen-Management (24 GB VRAM GPU)
Um Out-of-Memory-Fehler (OOM) bei langen Sequenzen auf einer 24 GB GPU (`mig_24gb:1`) zu verhindern und die effektive Batch-Größe konsistent auf $16$ zu halten, wurden die Batch-Größen und Gradient-Accumulation-Schritte angepasst:

| Token-Länge | SFT (`train_sft.py`) | DPO (`train_dpo.py`) | Effektive Batch-Größe |
|---|---|---|---|
| **256 Tokens** | `batch_size=8`, `accum=2` | `batch_size=2`, `accum=8` | **16** |
| **500 Tokens** | `batch_size=4`, `accum=4` | `batch_size=2`, `accum=8` | **16** |
| **1000 Tokens**| `batch_size=2`, `accum=8` | `batch_size=1`, `accum=16`| **16** |

### 2.2 Skript-Anpassungen
1. **[`scripts/modeling/train_sft.py`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/modeling/train_sft.py) & [`scripts/modeling/generate_dpo_dataset.py`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/modeling/generate_dpo_dataset.py)**:
   - Neuer CLI-Parameter `--reward_max_seq_len` (synchronisiert mit `--max_target_len`), um das vormals hardcodierte `[:150]` Token-Slicing während der Reward-Berechnung zu eliminieren.
2. **[`scripts/evaluation/evaluate_token_length_experiment.py`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/evaluation/evaluate_token_length_experiment.py)**:
   - Entfernung des hardcodierten Prompt-Präfixes (`--prompt_prefix ""`), um einen Trainings-Inferenz-Mismatch bei DPO zu verhindern.
   - Ergänzung einer vollständigen Evaluierungsroutine für die BiLSTM-Metrikmodelle auf dem Lebenshilfe-Goldstandard.

---

## 3. Ergebnisse Teil A: Evaluation der Simplicity-Metrik (BiLSTM MixUp)

Evaluation auf dem unabhängigen Lebenshilfe-Datensatz (`data/lebenshilfe/lebenshilfe_dataset_clean.json`, 37 Artikelpaare).

### 3.1 Statistische Kennzahlen

| Modell | Max Seq Len | $\varnothing$ Lambda (LS) | $\varnothing$ Lambda (AS) | Separation Margin ($\Delta$) | Accuracy (Schwelle 0.5) | Balanced Acc | MAE (Target 1/0) | Accuracy ($LS > AS$) |
|---|---|---|---|---|---|---|---|---|
| **`metric_mixup_256`** | 256 | $0.7744 \pm 0.104$ | $0.1136 \pm 0.073$ | $0.6608$ | $97.30\,\%$ | $97.30\,\%$ | $0.1696$ | $97.30\,\%$ |
| **`metric_mixup_500`** | 500 | $0.7557 \pm 0.128$ | $0.1363 \pm 0.088$ | $0.6194$ | $98.65\,\%$ | $98.65\,\%$ | $0.1903$ | $97.30\,\%$ |
| **`metric_mixup_1000`**| 1000 | $\mathbf{0.7788 \pm 0.128}$ | $\mathbf{0.1160 \pm 0.100}$ | $\mathbf{0.6628}$ 🏆 | $\mathbf{98.65\,\%}$ 🏆 | $\mathbf{98.65\,\%}$ 🏆 | $\mathbf{0.1686}$ 🏆 | $\mathbf{100.00\,\%}$ 🏆 |

### 3.2 Stratifizierte Trennschärfe nach Quelltext-Länge

| Modell | Kurz ($< 200$ Tokens) | Mittel ($200 - 450$ Tokens) | Lang ($> 450$ Tokens) | Spearman $r$ (AS / Länge) |
|---|---|---|---|---|
| **`metric_mixup_256`** | $0.5885$ | $0.6954$ | $0.6497$ | $-0.0072$ |
| **`metric_mixup_500`** | $0.4792$ | $0.6072$ | $0.6470$ | $-0.4211$ |
| **`metric_mixup_1000`**| $0.4795$ | $0.6616$ | $\mathbf{0.6897}$ 🏆 | $\mathbf{-0.5323}$ 🏆 |

### 3.3 Visualisierung: KDE-Dichteverteilungs-Plots
Gespeichert unter: `results/plots/compare_token_length_metrics_lh_kde.png`

* **Bimodale Verteilung**: Alle drei Modelle trennen Leichte Sprache (Grün, Peak bei $\approx 0.85$) und Alltagssprache (Blau, Peak bei $\approx 0.08$) scharf an der Schwelle $\lambda = 0.5$.
* **Vorteil 1000 Tokens**: Das 1000-Token-Modell eliminiert die bei 256 Tokens sichtbare Ausreißer-Häufung bei $\lambda \approx 0.45$, da lange Alltagstexte vollständig erfasst und nicht vorzeitig abgeschnitten werden.

---

## 4. Ergebnisse Teil B: Evaluation der Übersetzungsmodelle (SFT & DPO)

Evaluation auf dem Lebenshilfe-Testset mit 37 vollständigen Artikeln.

| Modell | Max Tokens | Simplicity ($R_{style}$) | Semantik ($R_{sem, AS}$) | Sim. Ref ($Sim_{ref}$) | **Composite Reward** | **BLEU** | **ROUGE-L F1** | **Avg. Tokens** | **Truncation Rate** |
|---|---|---|---|---|---|---|---|---|---|
| **`sft_len256`** | 256 | $0.6086$ | $\mathbf{0.8892}$ | $0.8649$ | $0.7489$ | $0.0089$ | $0.1292$ | $154.6$ | $43.2\,\%$ |
| **`sft_len500`** | 500 | $\mathbf{0.6583}$ | $0.8533$ | $0.8520$ | $\mathbf{0.7558}$ 🏆 | $0.0198$ | $0.1493$ | $237.6$ | $43.2\,\%$ |
| **`sft_len1000`**| 1000 | $0.5804$ | $0.8747$ | $\mathbf{0.8626}$ | $0.7276$ | $\mathbf{0.0249}$ 🏆 | $\mathbf{0.1606}$ 🏆 | $\mathbf{288.9}$ 🏆 | $\mathbf{18.9\,\%}$ 🏆 |
| `dpo_len256` | 256 | $0.4391$ | $0.8978$ | $0.8353$ | $0.6684$ | $0.0000$ | $0.0411$ | $37.8$ | $40.5\,\%$ |
| `dpo_len500` | 500 | $0.4175$ | $0.8840$ | $0.8233$ | $0.6507$ | $0.0009$ | $0.0422$ | $43.2$ | $29.7\,\%$ |
| `dpo_len1000`| 1000 | $0.3376$ | $0.8949$ | $0.8277$ | $0.6163$ | $0.0024$ | $0.0651$ | $114.1$ | $\mathbf{10.8\,\%}$ |

---

## 5. Zentrale Thesen & Erkenntnisse für die Masterarbeit

### 1. Drastische Reduktion von Truncation bei SFT-1000
Bei $256$ und $500$ Tokens brechen $43.2\,\%$ aller SFT-Übersetzungen mitten im Satz ab, weil das Tokenbudget für die ausführlichen Erklärungsstrukturen Leichter Sprache nicht ausreicht. Bei **1000 Tokens sinkt die Truncation-Rate auf $18.9\,\%$** (über $81\,\%$ aller Texte schließen syntaktisch sauber ab).

### 2. Lexikalische Qualität vervielfacht sich (BLEU & ROUGE)
* **BLEU-Score**: Steigt von $0.0089$ ($256$) über $0.0198$ ($500$) auf **$0.0249$ ($1000$)** — eine Steigerung um **$+180\,\%$** gegenüber dem 256er-Baseline-Modell.
* **ROUGE-L F1**: Steigt von $12.9\,\%$ auf **$16.1\,\%$**.
* **Ursache**: Mit 1000 Tokens kann das Modell auch Folgeabsätze übersetzen, anstatt nach der Einleitung zu stoppen.

### 3. Sweet-Spot-Differenzierung
* **500 Tokens**: Bester Kompromiss für abschnittsweise/paragraphenweise Vereinfachungen mit dem **höchsten Simplicity-Score ($0.6583$)** und **höchsten Composite Reward ($0.7558$)**.
* **1000 Tokens**: Unverzichtbar für die **vollständige Übersetzung ganzer Artikel**, da Textverlust und Satzabbrüche minimiert werden.

### 4. Funktionsweise der BiLSTM-Metrik bei 1000 Tokens
Entgegen theoretischer Bedenken hinsichtlich Gradient Degradation bei RNNs über 1000 Zeitschritte erzielt `metric_mixup_1000` die beste Performance ($100\,\%$ Klassifikationsgenauigkeit):
* Da es sich um eine **globale Stil-Klassifikation** handelt (und keine autoregressive Generierung), agiert das BiLSTM als kontinuierlicher Akkumulator stilistischer Merkmale.
* Durch die bidirektionale Verknüpfung des Vorwärts- und Rückwärtszustands bleiben Anfang und Ende des Textes gleichermaßen präsent.

### 5. DPO-Dynamik bei unterschiedlichen Token-Längen
DPO-Modelle neigen bei $w_{sem} = 0.5$ zu einer starken Kondensierung (38–114 Tokens) und selektieren bevorzugt extraktive Teilsätze, um den SBERT-Score zu maximieren. Für künftige DPO-Trainings empfiehlt sich eine höhere Gewichtung des Style-Scores ($w_{style} = 0.7$, $w_{sem} = 0.3$).

---

## 6. Zugehörige Skripte und Artefakte

- **SBatch-Suite**: [`scripts/sbatch/experiments/token_length/`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/experiments/token_length/)
- **Evaluationsskript**: [`scripts/evaluation/evaluate_token_length_experiment.py`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/evaluation/evaluate_token_length_experiment.py)
- **Metrik-Notebook**: [`notebooks/research/metric/compare_token_lengths.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/research/metric/compare_token_lengths.ipynb)
- **Übersetzungs-Notebook**: [`notebooks/research/translation/compare_token_lengths.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/research/translation/compare_token_lengths.ipynb)
- **KDE-Plot**: [`results/plots/compare_token_length_metrics_lh_kde.png`](file:///Users/fietescheel/Documents/Master%20Thesis/results/plots/compare_token_length_metrics_lh_kde.png)
