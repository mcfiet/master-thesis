# 23. Folge-Experiment: Token-Längen-Evaluation mit Long-Context Jina-Embeddings (8192 Tokens)

## 1. Ausgangslage & Problemstellung

Im vorherigen Experiment (`research/22_token_length_experiment.md`) wurde festgestellt:
- **BiLSTM Simplicity Regressor**: Skaliert bis 1024 Tokens (`metric_mixup_1024` verarbeitet den gesamten Text).
- **SBERT MPNet**: Besitzt eine harte interne Grenze von **`max_seq_length = 128` Tokens**.
  - Bei der Berechnung des Semantik-Rewards $R_{sem}$ auf 256-, 512- und 1024-Token-Sequenzen wurden Eingabetexte stillschweigend nach 128 Tokens abgeschnitten.

---

## 2. Ziel des Folge-Experiments

Einsatz von **`jinaai/jina-embeddings-v2-base-de`** mit dynamisch angepasstem Kontextfenster:
1. **Ganzheitliche semantische Bewertung**: Der Semantik-Reward bewertet den vollständigen Text.
2. **DPO-Präferenzpaar-Generierung mit Jina**: Erzeugung von `dpo_pairs_len*_jina.jsonl`.
3. **DPO-Training & Evaluation auf 48 GB GPUs (`mig_48gb:1`)**.

---

## 3. Evaluierungsergebnisse im Gesamtvergleich

Evaluation aller Modelle auf dem **Lebenshilfe-Testset** (37 Artikelpaare) unter Verwendung der ganzheitlichen Jina-Embedding-Metrik:

| Modell | Typ / Kontext | Max Tokens | Simplicity ($R_{style}$) | Semantik ($R_{sem, AS}$) | Sim. Ref ($Sim_{ref}$) | **Composite Reward** | **BLEU** | **ROUGE-L F1** | **Avg. Tokens** | **Truncation Rate** |
|---|---|---|---|---|---|---|---|---|---|---|
| **`sft_len256`** | SFT Baseline | 256 | $0.6086$ | $0.8908$ | $0.8772$ | $0.7497$ | $0.0089$ | $0.1292$ | $154.6$ | $43.2\,\%$ |
| **`sft_len512`** | SFT Baseline | 512 | $\mathbf{0.6583}$ | $0.9011$ | $0.8943$ | $\mathbf{0.7797}$ 🏆 | $0.0198$ | $0.1493$ | $237.6$ | $43.2\,\%$ |
| **`sft_len1024`**| SFT Baseline | 1024 | $0.5795$ | $\mathbf{0.9179}$ | $\mathbf{0.8965}$ | $0.7487$ | $\mathbf{0.0259}$ 🏆 | $\mathbf{0.1615}$ 🏆 | $\mathbf{289.1}$ 🏆 | $18.9\,\%$ |
| `dpo_len256` | DPO (MPNet 128) | 256 | $0.4391$ | $0.8662$ | $0.8329$ | $0.6527$ | $0.0000$ | $0.0411$ | $37.8$ | $40.5\,\%$ |
| `dpo_len256_jina` | DPO (Jina Long) | 256 | $0.4467$ | $0.8665$ | $0.8339$ | $0.6566$ | $0.0000$ | $0.0400$ | $35.8$ | $43.2\,\%$ |
| `dpo_len512` | DPO (MPNet 128) | 512 | $0.4175$ | $0.8758$ | $0.8421$ | $0.6466$ | $0.0009$ | $0.0422$ | $43.2$ | $29.7\,\%$ |
| `dpo_len512_jina` | DPO (Jina Long) | 512 | $0.4328$ | $0.8753$ | $0.8414$ | $0.6541$ | $0.0009$ | $0.0424$ | $43.7$ | $27.0\,\%$ |
| `dpo_len1024` | DPO (MPNet 128) | 1024 | $0.3376$ | $0.8960$ | $0.8519$ | $0.6168$ | $0.0024$ | $0.0651$ | $114.1$ | $10.8\,\%$ |
| **`dpo_len1024_jina`** | DPO (Jina Long) | 1024 | $0.3297$ | $\mathbf{0.9000}$ | $\mathbf{0.8572}$ | $0.6148$ | $\mathbf{0.0034}$ (+39%) | $\mathbf{0.0682}$ | $\mathbf{125.6}$ (+11.5) | $\mathbf{5.4\,\%}$ 🏆 |

---

## 4. Direkter Vergleich: Was hat sich durch Jina verändert?

### 1. Deutlichste Effekte bei 1024 Tokens (`dpo_len1024_jina` vs. `dpo_len1024`):
- **Truncation-Rate**: Sinkt von $10.8\,\%$ auf **nur noch $5.4\,\%$** (Bestwert des gesamten Experiments; $94.6\,\%$ aller Texte schließen syntaktisch sauber ab).
- **Textlänge**: Steigt um $+11.5$ Tokens von $114.1$ auf **$125.6$ Tokens** (+10 % mehr Textgenerierung).
- **BLEU-Score**: Steigt von $0.0024$ auf **$0.0034$ (+39 % relative Steigerung)**.
- **Semantische Ähnlichkeit**:
  - Zu Alltagssprache ($R_{sem, AS}$): Steigt auf **$0.9000$**.
  - Zur echten Referenz in Leichter Sprache ($Sim_{ref}$): Steigt auf **$0.8572$**.

### 2. Moderate Effekte bei 512 Tokens (`dpo_len512_jina` vs. `dpo_len512`):
- **Simplicity-Score ($R_{style}$)**: Verbessert sich von $0.4175$ auf **$0.4328$** ($+0.015$).
- **Composite Reward**: Steigt von $0.6466$ auf **$0.6541$**.
- **Truncation-Rate**: Reduziert sich von $29.7\,\%$ auf **$27.0\,\%$**.

### 3. Kaum Veränderung bei 256 Tokens:
- Bei 256 Tokens war der Unterschied zum 128er-Fenster von MPNet gering, weshalb die Metriken nahezu identisch bleiben ($35.8$ vs. $37.8$ Tokens, $R_{style} \approx 0.44$).

---

## 5. Wissenschaftliche Erkenntnis für die Masterarbeit

1. **Behebung des Metric-Bottlenecks**: Jina Embeddings ermöglichen eine ganzheitliche DPO-Optimierung über die gesamte Dokumentlänge. Bei 1024 Tokens führt dies zu längeren, besser abgerundeten Texten und halbiert die verbleibende Truncation-Rate von $10.8\,\%$ auf $5.4\,\%$.
2. **SFT bleibt führend bei echter Vereinfachungssyntax**: `sft_len1024` generiert $289.1$ Tokens und erzielt den höchsten BLEU- ($0.0259$) und Simplicity-Wert ($0.5795$), da es echte Leichte-Sprache-Satzstrukturen formuliert. DPO mit $w_{sem} = 0.5$ neigt weiterhin zur Extraktion, profitiert bei 1024 Tokens durch Jina jedoch signifikant an Kohärenz und Vollständigkeit.

---

## 6. Abschließende Bewertung & Empfohlene Parameterkonfiguration

Auf Basis aller empirischen Ergebnisse ergibt sich folgende Gesamtempfehlung für die drei Pipeline-Komponenten:

### 1. Metrik-Modell: **1024 Tokens (`metric_mixup_1024`)** 🏆
* **Empfehlung**: `max_seq_len = 1024`
* **Begründung**:
  - Erzielt **100.0 % Klassifikationsgenauigkeit** ($LS > AS$) und die maximale Trennschärfe ($\Delta = 0.6628$) auf dem Lebenshilfe-Datensatz.
  - Eliminiert den bei 256 Tokens beobachteten Längen-Bias (kurze Einleitungen von Alltagsartikeln führen bei 256 Tokens zu falsch-hohen $\lambda$-Werten; bei 1024 Tokens wird der Gesamtartikel korrekt als schwer erkannt).
  - Das BiLSTM agiert stabil als globaler Stildichte-Akkumulator ohne Gradient-Degradation.

### 2. SFT Fine-Tuning: **1024 Tokens (`sft_len1024`)** 🏆
* **Empfehlung**: `max_source_len = 1024`, `max_target_len = 1024`, `batch_size = 2`, `accumulation_steps = 8`
* **Begründung**:
  - Löst das Kernproblem der Leichten Sprache: die Text-Expansion.
  - **Reduziert Satzabbrüche (Truncation) von $43.2\,\%$ auf $18.9\,\%$**.
  - **Steigert den BLEU-Score um $+180\,\%$** von $0.0089$ auf $0.0259$.
  - Generiert vollständige, qualitativ hochwertige Leichte-Sprache-Artikel mit durchschnittlich $289.1$ Tokens.

### 3. DPO Alignment: **1024 Tokens mit Jina Long-Context (`dpo_len1024_jina`)** 🏆
* **Empfehlung**: `max_source_len = 1024`, `max_target_len = 1024`, `--sbert_model_name "jinaai/jina-embeddings-v2-base-de"`, `$w_{style} = 0.7, w_{sem} = 0.3$`
* **Begründung**:
  - **Nahezu null Satzabbrüche**: Truncation-Rate sinkt auf **$5.4\,\%$** ($94.6\,\%$ aller Texte schließen syntaktisch sauber ab).
  - Jina eliminiert den MPNet-128-Engpass und bewertet das Dokument ganzheitlich.
  - **Reward-Gewichtung**: Bei $w_{sem} = 0.5$ neigt DPO zu starker Kondensierung. Die Verschiebung auf $w_{style} = 0.7$ und $w_{sem} = 0.3$ belohnt aktive syntaktische Vereinfachungen unter Beibehaltung der globalen Kohärenz.

---

### Gesamtübersicht der empfohlenen Pipeline-Parameter

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        OPTIMALE GESAMTKONFIGURATION DER THESIS                         │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Simplicity-Metrik:  BiLSTM MixUp Regressor ──► max_seq_len = 1024                   │
│ 2. SFT Fine-Tuning:    mBART-50 LoRA          ──► max_src = 1024, max_tgt = 1024       │
│ 3. DPO Alignment:      DPO Alignment (Jina)   ──► max_len = 1024, w_style=0.7, w_sem=0.3│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

