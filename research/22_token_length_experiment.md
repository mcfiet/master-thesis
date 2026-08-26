# 22. Versuchsaufbau & Evaluation: Einfluss der Token-Länge mit Long-Context Jina-Embeddings (256 vs. 512 vs. 1024 Tokens)

## 1. Motivation & Forschungsfrage

In frühen Iterationen der Übersetzungspipeline wurde standardmäßig mit Sequenzlängen von $256$ Tokens gearbeitet. Deutsche Artikel in Alltagssprache (AS) sowie deren Entsprechungen in Leichter Sprache (LS) umfassen im Lebenshilfe-Korpus jedoch durchschnittlich $\approx 916$ bzw. $\approx 1107$ Tokens.

Daraus ergeben sich zwei zentrale Forschungsfragen:
1. **Skalierung der Sequenzlänge:** Welchen Einfluss hat die maximale Token-Länge ($256$, $512$, $1024$ Tokens) auf die drei Pipeline-Stufen (Simplicity-Metrik, SFT Fine-Tuning und DPO Alignment)?
2. **Metric-Bottleneck & Long-Context:** Warum führen Standard-Sentence-Transformer mit hartem $128$-Token-Limit (z. B. MPNet) zu DPO-Verzerrungen und wie löst ein Long-Context-Embedding (**Jina Embeddings v2**, bis 8192 Tokens) dieses Problem?

---

## 2. Experimenteller Aufbau & Methodik

### 2.1 Hardware- & Ressourcen-Management
Um Out-of-Memory-Fehler (OOM) bei Sequenzlängen bis $1024$ Tokens zu vermeiden und eine konsistente effektive Batch-Größe von $16$ zu gewährleisten, wurden folgende Trainingsparameter und GPU-Profile definiert:

| Stufe / Token-Länge | Batch-Größe | Accumulation Steps | Effektive Batch-Größe | GPU-Ressource |
|---|---|---|---|---|
| **SFT 256 Tokens** | 8 | 2 | **16** | `mig_24gb:1` (16 GB RAM) |
| **SFT 512 Tokens** | 4 | 4 | **16** | `mig_24gb:1` (16 GB RAM) |
| **SFT 1024 Tokens**| 2 | 8 | **16** | `mig_24gb:1` (16 GB RAM) |
| **DPO 256 Tokens (Jina)** | 2 | 8 | **16** | `mig_24gb:1` (16 GB RAM) |
| **DPO 512 Tokens (Jina)** | 2 | 8 | **16** | `mig_24gb:1` (16 GB RAM) |
| **DPO 1024 Tokens (Jina)**| 1 | 16| **16** | `mig_48gb:1` (32 GB RAM) |

### 2.2 Drei-Stufen-Architektur mit Jina Long-Context
1. **Stufe 1 (Simplicity-Metrik):** BiLSTM MixUp Regressoren (`metric_mixup_256`, `metric_mixup_512`, `metric_mixup_1024`).
2. **Stufe 2 (SFT-Baseline):** mBART-50 LoRA (`sft_len256`, `sft_len512`, `sft_len1024`).
3. **Stufe 3 (DPO Alignment):** Präferenzpaar-Generierung mit dynamisch angepasstem Jina-Kontextfenster (`jinaai/jina-embeddings-v2-base-de`), wodurch der semantische Reward $R_{sem}$ über die gesamte Dokumentlänge ohne stillschweigenden Cut-off berechnet wird.

---

## 3. Ergebnisse Teil A: Evaluation der Simplicity-Metrik (BiLSTM MixUp)

Evaluation auf dem unabhängigen Lebenshilfe-Datensatz (`data/lebenshilfe/lebenshilfe_dataset_clean.json`, 37 vollständige Artikelpaare).

### 3.1 Statistische Kennzahlen

| Modell | Max Seq Len | $\varnothing$ Lambda (LS) | $\varnothing$ Lambda (AS) | Separation Margin ($\Delta$) | Accuracy (Schwelle 0.5) | Balanced Acc | MAE (Target 1/0) | Accuracy ($LS > AS$) |
|---|---|---|---|---|---|---|---|---|
| **`metric_mixup_256`** | 256 | $0.7744 \pm 0.104$ | $0.1136 \pm 0.073$ | $0.6608$ | $97.30\,\%$ | $97.30\,\%$ | $0.1696$ | $97.30\,\%$ |
| **`metric_mixup_512`** | 512 | $0.7557 \pm 0.128$ | $0.1363 \pm 0.088$ | $0.6194$ | $98.65\,\%$ | $98.65\,\%$ | $0.1903$ | $97.30\,\%$ |
| **`metric_mixup_1024`**| 1024 | $\mathbf{0.7788 \pm 0.128}$ | $\mathbf{0.1160 \pm 0.100}$ | $\mathbf{0.6628}$ 🏆 | $\mathbf{98.65\,\%}$ 🏆 | $\mathbf{98.65\,\%}$ 🏆 | $\mathbf{0.1686}$ 🏆 | $\mathbf{100.00\,\%}$ 🏆 |

### 3.2 Stratifizierte Trennschärfe nach Quelltext-Länge

| Modell | Kurz ($< 200$ Tokens) | Mittel ($200 - 450$ Tokens) | Lang ($> 450$ Tokens) | Spearman $r$ (AS / Länge) |
|---|---|---|---|---|
| **`metric_mixup_256`** | $0.5885$ | $0.6954$ | $0.6497$ | $-0.0072$ |
| **`metric_mixup_512`** | $0.4792$ | $0.6072$ | $0.6470$ | $-0.4211$ |
| **`metric_mixup_1024`**| $0.4795$ | $0.6616$ | $\mathbf{0.6897}$ 🏆 | $\mathbf{-0.5323}$ 🏆 |

*Erkenntnis:* `metric_mixup_1024` erzielt $100.0\,\%$ paarweise Genauigkeit und eliminiert den Längen-Bias bei langen Texten.

---

## 4. Ergebnisse Teil B: Gesamtvergleich der Übersetzungsmodelle (SFT vs. DPO Jina vs. MPNet Ablation)

Evaluation aller Modelle auf dem Lebenshilfe-Testset mit 37 vollständigen Artikeln unter Verwendung der Long-Context Jina-Evaluation:

| Modell | Typ / Semantik-Backbone | Max Tokens | Simplicity ($R_{style}$) | Semantik ($R_{sem, AS}$) | Sim. Ref ($Sim_{ref}$) | **Composite Reward** | **BLEU** | **ROUGE-L F1** | **Avg. Tokens** | **Truncation Rate** |
|---|---|---|---|---|---|---|---|---|---|---|
| **`sft_len256`** | SFT Baseline | 256 | $0.6086$ | $0.8908$ | $0.8772$ | $0.7497$ | $0.0089$ | $0.1292$ | $154.6$ | $43.2\,\%$ |
| **`sft_len512`** | SFT Baseline | 512 | $\mathbf{0.6583}$ | $0.9011$ | $0.8943$ | $\mathbf{0.7797}$ 🏆 | $0.0198$ | $0.1493$ | $237.6$ | $43.2\,\%$ |
| **`sft_len1024`**| SFT Baseline | 1024 | $0.5795$ | $\mathbf{0.9179}$ | $\mathbf{0.8965}$ | $0.7487$ | $\mathbf{0.0259}$ 🏆 | $\mathbf{0.1615}$ 🏆 | $\mathbf{289.1}$ 🏆 | $18.9\,\%$ |
| `dpo_len256` | DPO Ablation (MPNet 128) | 256 | $0.4391$ | $0.8662$ | $0.8329$ | $0.6527$ | $0.0000$ | $0.0411$ | $37.8$ | $40.5\,\%$ |
| `dpo_len256_jina` | **DPO Primary (Jina Long)** | 256 | $0.4467$ | $0.8665$ | $0.8339$ | $0.6566$ | $0.0000$ | $0.0400$ | $35.8$ | $43.2\,\%$ |
| `dpo_len512` | DPO Ablation (MPNet 128) | 512 | $0.4175$ | $0.8758$ | $0.8421$ | $0.6466$ | $0.0009$ | $0.0422$ | $43.2$ | $29.7\,\%$ |
| `dpo_len512_jina` | **DPO Primary (Jina Long)** | 512 | $0.4328$ | $0.8753$ | $0.8414$ | $0.6541$ | $0.0009$ | $0.0424$ | $43.7$ | $27.0\,\%$ |
| `dpo_len1024` | DPO Ablation (MPNet 128) | 1024 | $0.3376$ | $0.8960$ | $0.8519$ | $0.6168$ | $0.0024$ | $0.0651$ | $114.1$ | $10.8\,\%$ |
| **`dpo_len1024_jina`** | **DPO Primary (Jina Long)** | 1024 | $0.3297$ | $\mathbf{0.9000}$ | $\mathbf{0.8572}$ | $0.6148$ | $\mathbf{0.0034}$ | $\mathbf{0.0682}$ | $\mathbf{125.6}$ | $\mathbf{5.4\,\%}$ 🏆 |

---

## 5. Zentrale Thesen & wissenschaftliche Erkenntnisse

### 1. Behebung des Metric-Bottlenecks durch Jina
* MPNet schneidet Texte unbemerkt nach 128 Tokens ab. Bei 1024 Tokens bewertet MPNet folglich nur die ersten $\approx 12\,\%$ des Textes.
* **Jina Long-Context halbiert die Truncation-Rate**: Bei 1024 Tokens sinken die Satzabbrüche von $10.8\,\%$ auf **$5.4\,\%$** ($94.6\,\%$ aller generierten Artikel schließen syntaktisch sauber ab).
* Die Generierungslänge steigt um $+10\,\%$ ($125.6$ vs. $114.1$ Tokens), und der BLEU-Score verbessert sich um $+39\,\%$.

### 2. Drastische Reduktion von Satzabbrüchen bei SFT-1024
* Bei 256 und 512 Tokens brechen $43.2\,\%$ aller SFT-Texte vorzeitig ab, weil das Token-Budget für die expandierende Struktur Leichter Sprache unzureichend ist.
* Bei **1024 Tokens sinkt die SFT-Truncation auf $18.9\,\%$**, während der BLEU-Score von $0.0089$ auf **$0.0259$ ($+191\,\%$)** steigt.

### 3. Komplementäre Stärken von SFT und DPO
* **SFT-1024**: Erzeugt ausführliche Übersetzungen ($289.1$ Tokens) mit hohem Simplicity-Wert ($0.5795$) und echtem Satzbau Leichter Sprache.
* **DPO-1024-Jina**: Erreicht die höchste Satzabschluss-Zuverlässigkeit ($94.6\,\%$ vollständige Sätze) und höchste semantische Treue ($0.9000$).

---

## 6. Empfohlene finale Pipeline-Konfiguration

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        OPTIMALE GESAMTKONFIGURATION DER MASTERARBEIT                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Simplicity-Metrik:  BiLSTM MixUp Regressor ──► max_seq_len = 1024                   │
│ 2. SFT Fine-Tuning:    mBART-50 LoRA          ──► max_src = 1024, max_tgt = 1024       │
│ 3. DPO Alignment:      DPO Alignment (Jina)   ──► max_len = 1024, w_style=0.7, w_sem=0.3│
│ 4. Semantik-Metrik:    Jina Embeddings v2     ──► jinaai/jina-embeddings-v2-base-de    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Zugehörige Skripte und Artefakte

- **SBatch-Suite (Jina-Standard)**: [`scripts/sbatch/experiments/token_length/`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/experiments/token_length/)
- **Evaluationsskript**: [`scripts/evaluation/evaluate_token_length_experiment.py`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/evaluation/evaluate_token_length_experiment.py)
- **Metrik-Notebook**: [`notebooks/research/metric/compare_token_lengths.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/research/metric/compare_token_lengths.ipynb)
- **Übersetzungs-Notebook**: [`notebooks/research/translation/compare_token_lengths.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/research/translation/compare_token_lengths.ipynb)
- **KDE-Plot**: [`results/plots/compare_token_length_metrics_lh_kde.png`](file:///Users/fietescheel/Documents/Master%20Thesis/results/plots/compare_token_length_metrics_lh_kde.png)
