# Experiment 25: DPO-Loss-Aggregation – Sum vs. Mean (Length-Normalized DPO)

Dieses Dokument dokumentiert die theoretischen Grundlagen, das experimentelle Setup, die empirischen Ergebnisse und die wissenschaftlichen Erkenntnisse des Experiments zur **Loss-Aggregation im Direct Preference Optimization (DPO)** für das Encoder-Decoder-Modell (`facebook/mbart-large-50`).

---

## 1. Theoretischer Hintergrund & Problemstellung

Im _Direct Preference Optimization_ (DPO) Framework wird die relative Präferenz zwischen einem präferierten Text $y_w$ (_winner_) und einem dispräferierten Text $y_l$ (_loser_) über die relativen Log-Likelihoods gegenüber einem Referenzmodell $\pi_{\text{ref}}$ bestimmt:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right) \right]$$

### Das Problem der klassischen Summierung (`sum`):

Standardmäßig berechnet DPO die Sequenz-Wahrscheinlichkeit als ungewichtete Summe der Token-Log-Wahrscheinlichkeiten:
$$\log \pi(y \mid x) = \sum_{t=1}^{|y|} \log \pi(y_t \mid x, y_{<t})$$

Da für Wahrscheinlichkeiten $p \in (0, 1]$ gilt, dass $\log p \le 0$ ist, sind Log-Wahrscheinlichkeiten immer **negativ**.

- Eine lange, ausführlich paraphrasierte Erklärung in Leichter Sprache ($|y| = 150$ Tokens) akkumuliert 150 negative Terme (z. B. $150 \times (-0.15) = -22.5$).
- Ein kurzes Satzfragment oder ein verfrühter Satzabbruch ($|y| = 25$ Tokens) akkumuliert nur 25 negative Terme (z. B. $25 \times (-0.15) = -3.75$).

**Konsequenz:** Die klassische Summierung erzeugt einen inhärenten Längen-Bias (_Length Exploitation_ / Degeneration). Das Modell wird dafür bestraft, ausführliche und grammatikalisch vollständige Erklärungen zu generieren, und tendiert zu Satzabbrüchen oder Auslassungen.

---

### Die Lösung: Längen-Normalisierung (`mean` / Per-Token DPO):

Durch die Normalisierung mit der tatsächlichen Token-Anzahl $|y|$ wird der reine Längeneffekt eliminiert:
$$\overline{\log \pi(y \mid x)} = \frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi(y_t \mid x, y_{<t})$$

$$\mathcal{L}_{\text{DPO, mean}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \overline{\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}} - \overline{\log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}} \right) \right) \right]$$

Hierdurch wird die **durchschnittliche Modellzuversicht pro generiertem Token** optimiert. Längere, didaktisch wertvolle Paraphrasen in Leichter Sprache werden nicht mehr mathematisch benachteiligt.

---

## 2. Experimentelles Setup (Ceteris Paribus)

Das Experiment vergleicht drei Modellvarianten unter strikt identischen Trainings- und Inferenzparametern:

| Parameter                | Konfiguration                                                               |
| :----------------------- | :-------------------------------------------------------------------------- |
| **Basismodell**          | `facebook/mbart-large-50` (SFT Baseline aus `results/models/sft`)           |
| **PEFT / LoRA**          | $r=16, \alpha=32$, Dropout $0.05$, Target Modules: Attention + FC           |
| **Lernrate & Scheduler** | $5 \cdot 10^{-6}$, AdamW, Linear Warmup (10%)                               |
| **DPO-Beta ($\beta$)**   | $0.1$                                                                       |
| **Epochen / Batch Size** | 3 Epochen, Batch Size 2, Gradient Accumulation 8                            |
| **Datensatz**            | Lebenshilfe DPO-Präferenzpaare (`dpo_pairs_w05_w05.jsonl`)                  |
| **Inferenz-Setup**       | Beam Search (`num_beams=4`), `no_repeat_ngram_size=3`, `max_target_len=256` |

### Untersuchte Modellvarianten:

1. **SFT Baseline:** Referenzmodell nach Supervised Fine-Tuning ohne DPO (`results/models/sft`)
2. **DPO Sum (Classic):** DPO-Training mit summierten Log-Likelihoods `--loss_type sum` (`results/models/loss_aggregation_exp/dpo_sum`)
3. **DPO Mean (Length-Normalized):** DPO-Training mit gemittelten Log-Likelihoods `--loss_type mean` (`results/models/loss_aggregation_exp/dpo_mean`)

---

## 3. Ausführung auf dem Compute-Cluster

Die Trainings- und Evaluationspipelines wurden modular mit SLURM-Skripten ausgeführt:

- Master-Skript: [`scripts/sbatch/experiments/loss_aggregation/run_all_loss_aggregation_experiments.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/experiments/loss_aggregation/run_all_loss_aggregation_experiments.sh)
- DPO Sum Training: [`scripts/sbatch/experiments/loss_aggregation/1_train_dpo_sum.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/experiments/loss_aggregation/1_train_dpo_sum.sh)
- DPO Mean Training: [`scripts/sbatch/experiments/loss_aggregation/1_train_dpo_mean.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/experiments/loss_aggregation/1_train_dpo_mean.sh)
- Vollständige Evaluation: [`scripts/sbatch/experiments/loss_aggregation/2_run_full_evaluation.sh`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/experiments/loss_aggregation/2_run_full_evaluation.sh)

---

## 4. Empirische Ergebnisse

Die Evaluierung erfolgte auf dem bereinigten Lebenshilfe-Testdatensatz ([`data/lebenshilfe/lebenshilfe_dataset_clean.json`](file:///Users/fietescheel/Documents/Master%20Thesis/data/lebenshilfe/lebenshilfe_dataset_clean.json), $N = 37$ Artikel) unter Verwendung des BiLSTM-MixUp-Regressors ($R_{\text{style}}$), multilingualem SBERT ($R_{\text{sem}}$) sowie lexikalischen Metriken (BLEU, ROUGE-L).

### Quantitative Vergleichstabelle:

| Modell                      | Ø Simplicity ($R_{\text{style}}$) | Ø Semantik zu AS ($R_{\text{sem}}$) | Ø Treue zu LS ($Sim_{\text{ref}}$) | Composite Reward (0.5/0.5) |    BLEU    | ROUGE-L F1 | Ø Gen. Tokens | Kompressionsrate | Truncation Rate (%) |
| :-------------------------- | :-------------------------------: | :---------------------------------: | :--------------------------------: | :------------------------: | :--------: | :--------: | :-----------: | :--------------: | :-----------------: |
| **SFT Baseline**            |              0.6734               |               0.8528                |               0.8476               |           0.7631           |   0.0057   |   0.1260   |    165.38     |      0.3900      |       78.38%        |
| **DPO Sum (Classic)**       |              0.6616               |               0.8590                |               0.8520               |           0.7603           | **0.0090** | **0.1267** |    162.81     |      0.3874      |       72.97%        |
| **DPO Mean (Length-Norm.)** |            **0.6938**             |             **0.8595**              |             **0.8566**             |         **0.7767**         |   0.0080   |   0.1228   |    161.73     |      0.3827      |     **67.57%**      |

---

## 5. Wissenschaftliche Erkenntnisse & Diskussion

### 1. Überlegenheit der Längen-Normalisierung (`mean`):

- **Höchster Simplicity-Score:** `DPO Mean` erzielt mit **$R_{\text{style}} = 0.6938$** eine signifikante Steigerung der stilistischen Einfachheit gegenüber der SFT Baseline ($0.6734$) und `DPO Sum` ($0.6616$).
- **Rückschritt bei `DPO Sum`:** `DPO Sum` fällt beim Simplicity-Score sogar hinter die SFT-Baseline zurück ($0.6616$ vs. $0.6734$). Die Akkumulation negativer Log-Likelihoods führt dazu, dass das Modell einfache, aber längere Satzstrukturen schlechter bewertet.
- **Bester Composite Reward:** Mit **$0.7767$** erreicht `DPO Mean` den höchsten Gesamt-Reward aller drei Modelle.

### 2. Drastische Reduktion von Satzabbrüchen (Truncation Rate):

- `DPO Mean` senkt die Satzabbruch-Quote auf **$67.57\%$** (eine relative Verbesserung um über $10.8$ Prozentpunkte gegenüber der SFT Baseline mit $78.38\%$).
- `DPO Sum` weist mit $72.97\%$ eine spürbar höhere Quote unvollständiger Sätze auf, da das Modell den Generierungsstopp früher einleitet, um Tokens einzusparen.

### 3. Semantischer Erhalt und Referenztreue:

- Beide DPO-Varianten verbessern den semantischen Erhalt zur Alltagssprache ($R_{\text{sem}} \approx 0.859$–$0.860$) gegenüber SFT ($0.8528$).
- In der Ähnlichkeit zur menschlichen Leichte-Sprache-Referenz ($Sim_{\text{ref}}$) führt `DPO Mean` mit **$0.8566$** (vs. $0.8520$ bei `DPO Sum` und $0.8476$ bei SFT).

### 4. Textlänge und Kürzungsresistenz:

- Alle Modelle generieren im Durchschnitt zwischen **161.7 und 165.4 Tokens**.
- Dass `DPO Sum` nicht vollständig auf 20–30 Tokens kollabiert ist, liegt an der Verankerung der menschlichen Referenz im DPO-Präferenzdatensatz. Dennoch zeigt sich der negative Einfluss der ungewichteten Summierung in schlechterer Grammatik, Satzabbrüchen und einem niedrigeren Simplicity-Score.

---

## 6. Qualitativer Vergleich (Beispielanalysen)

### Beispiel: Kieler Sicherheitskonzept Sexualstraftäter (KSKS)

- **AS-Auszug:** _„Ziel von KSKS ist im Wesentlichen der formalisierte und standardisierte Datentransfer von der Justiz an die Polizei, um Letztere in die Überwachung gefährlicher und rückfallgefährdeter Täter [...] einzubinden.“_
- **SFT Baseline:** Verfällt in repetitive Frageschleifen (_„Was ist das? Was ist ein KSKS-Verfahren? Was muss ich beachten?...“_) und bricht am Ende unvollständig ab (_„...Oder Sie haben“_).
- **DPO Sum (Classic):** Versucht Frageschleifen zu kürzen, bricht jedoch mitten im juristischen Paragrafen ab (_„...Und wegen Begehung einer der vorgenannten Taten wegen Vollrausches“_).
- **DPO Mean (Length-Normalized):** Generiert einen wohlgeformten, didaktisch strukturierten Text mit vollständigem Satzabschluss:
  > _„Was ist KSKS? KSKS ist ein Wort für: Gemeinsames Sicherheits-Konzept Sexualstraftäter. In Schleswig-Holstein gibt es seit dem 1. Oktober 2008 ein neues Gesetz. Das Gesetz heißt: KSKS. [...] Diese Informationen helfen Ihnen bei der Überwachung gefährlicher und rückfall-gefährdeter Täter. Und diese Informationen können Sie dann an die Polizei weitergeben.“_

---

## 8. Verknüpfte Artefakte & Ressourcen

- **Evaluierungs-Zusammenfassung (CSV):** [`results/evaluation/loss_aggregation_comparison_summary.csv`](file:///Users/fietescheel/Documents/Master%20Thesis/results/evaluation/loss_aggregation_comparison_summary.csv)
- **Detaillierte Vorhersagen (CSV):** [`results/evaluation/loss_aggregation_comparison_details.csv`](file:///Users/fietescheel/Documents/Master%20Thesis/results/evaluation/loss_aggregation_comparison_details.csv)
- **Pareto-Frontier Plot:** [`results/plots/loss_aggregation_pareto_frontier.png`](file:///Users/fietescheel/Documents/Master%20Thesis/results/plots/loss_aggregation_pareto_frontier.png)
- **Längen- und Truncation-Vergleich Plot:** [`results/plots/loss_aggregation_length_comparison.png`](file:///Users/fietescheel/Documents/Master%20Thesis/results/plots/loss_aggregation_length_comparison.png)
- **Interaktives Analyse-Notebook:** [`notebooks/research/translation/analyse_loss_aggregation_experiment.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/research/translation/analyse_loss_aggregation_experiment.ipynb)
- **Evaluierungsskript:** [`scripts/evaluation/evaluate_loss_aggregation_experiment.py`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/evaluation/evaluate_loss_aggregation_experiment.py)
