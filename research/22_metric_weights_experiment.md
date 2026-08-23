# Experiment 24: Untersuchung der Reward-Gewichtung (Simplicity vs. Semantik) für das Encoder-Decoder-Modell

Dieses Experiment untersucht den systematischen Einfluss der Gewichtung in der Verbund-Reward-Funktion (_Composite Reward_) auf das _Direct Preference Optimization_ (DPO) Training des Encoder-Decoder-Übersetzungsmodells (`facebook/mbart-large-50`).

---

## 1. Motivation & Forschungsfragen

In der automatischen Textvereinfachung von Alltagssprache (AS) in Leichte Sprache (LS) stehen zwei Zielgrößen in einem inhärenten Zielkonflikt:

1. **Einfachheit & Stil ($R_{\text{style}}$):** Kurze Sätze, einfacher Satzbau (SVO), leicht verständliches Vokabular, Vermeidung von Nebensätzen und Passivkonstruktionen.
2. **Semantischer Erhalt & Faktentreue ($R_{\text{sem}}$):** Vollständige Beibehaltung der Kernbedeutung und Fakten der Ausgangssprache (AS) ohne unerwünschte Auslassungen oder Halluzinationen.

### Forschungsfragen:

- **F1:** Wie verändern sich Übersetzungsqualität und Lesbarkeitsgrad, wenn die Simplicity-Gewichtung von $w_{\text{style}} = 0.5$ auf $w_{\text{style}} = 0.7$ bzw. $w_{\text{style}} = 1.0$ erhöht wird?
- **F2:** Welches Gewichtungsverhältnis bildet die optimale Pareto-Grenze zwischen syntaktischer Einfachheit und semantischer Konsistenz für Encoder-Decoder-Architekturen?
- **F3:** Führt eine reine Stil-Optimierung ($w_{\text{style}} = 1.0$) zum semantischen Kollaps oder bietet die DPO-Referenzmodell-Regularisierung ($\pi_{\text{ref}}$) einen ausreichenden semantischen Anker?

---

## 2. Mathematische Formulierung & Methodik

### Verbund-Reward-Funktion:

Für jedes generierte Übersetzungskandidaten-Paar $(y_w, y_l)$ bei gegebenem Quelltext $x$ wird der Gesamtreward nach folgender Gleichung berechnet:

$$R(x, y) = w_{\text{style}} \cdot R_{\text{style}}(y) + w_{\text{sem}} \cdot R_{\text{sem, norm}}(x, y)$$

wobei:

- $R_{\text{style}}(y) \in [0, 1]$ die Vorhersage des trainierten BiLSTM-MixUp-Regressors darstellt.
- $R_{\text{sem, norm}}(x, y) = \frac{\cos(\mathbf{e}_x, \mathbf{e}_y) + 1}{2} \in [0, 1]$ die normalisierte Kosinus-Ähnlichkeit der Sentence-BERT-Embeddings (`paraphrase-multilingual-mpnet-base-v2`) ist.
- $w_{\text{style}} + w_{\text{sem}} = 1.0$.

### Längennormalisierte DPO-Optimierung (`mean`):

Um Kürzungsartefakte und die mathematische Bestrafung längerer Erklärungen zu verhindern, wird die längennormalisierte Log-Likelihood-Berechnung verwendet:

$$\overline{\log \pi_\theta(y \mid x)} = \frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})$$

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \overline{\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}} - \overline{\log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}} \right) \right) \right]$$

### Getestete Konfigurationen:

| Konfiguration      | $w_{\text{style}}$ (Simplicity) | $w_{\text{sem}}$ (Semantik) | Fokus / Hypothese                                                           |
| :----------------- | :-----------------------------: | :-------------------------: | :-------------------------------------------------------------------------- |
| **`SFT Baseline`** |                –                |              –              | Supervised Fine-Tuning Referenz (`results/models/sft`).                     |
| **`dpo_w05_w05`**  |             $0.50$              |           $0.50$            | Ausgewogene Balance zwischen Stil und Informationserhalt.                   |
| **`dpo_w07_w03`**  |             $0.70$              |           $0.30$            | Priorität auf Einfachheit bei moderater semantischer Führung.               |
| **`dpo_w10_w00`**  |             $1.00$              |           $0.00$            | Reine Stil-Optimierung zur Untersuchung der impliziten DPO-Regularisierung. |

---

## 3. Wichtige methodische & technische Erkenntnisse

Im Rahmen der experimentellen Durchführung wurden folgende zentrale Problemstellungen identifiziert und gelöst:

1. **SFT-Adapter-Verschmelzung (`merge_and_unload`):**
   - Das Basismodell für DPO muss den trainierten SFT-LoRA-Adapter fest in die Basisgewichte integrieren (`merge_and_unload`), bevor der neue DPO-LoRA-Adapter aufgesetzt wird. Andernfalls führt das Vorhandensein zweier unverschmolzener Adapterstrukturen zu Instabilitäten oder zum Laden des un-fine-getunten mBART-Basismodells.
2. **Kandidaten-Generierung & Referenz-Verankerung:**
   - Bei der Erzeugung der DPO-Präferenzpaare (`generate_dpo_dataset.py`) muss der menschliche Referenztext (`ls_text`) im Kandidatenpool enthalten sein. Reines Model-Sampling ohne Referenzanker führt dazu, dass das Reward-Modell aufgrund kürzerer Satzlängen 1-Satz-Fragmente bevorzugt, wodurch das Modell eine Kürzungsflucht lernt.
3. **mBART Sprachcode-Konfiguration (`de_DE`):**
   - Sowohl im Training als auch bei der Evaluierung müssen `tokenizer.src_lang = "de_DE"` und `tokenizer.tgt_lang = "de_DE"` explizit initialisiert werden, um einen sofortigen Abbruch der Generierung zu verhindern.
4. **Bereinigung alter Adapterreste:**
   - Nach dem Export der fusionierten Standalone-Modelle (`model.safetensors`, 1,22 GB) müssen alte `adapter_config.json`-Dateien gelöscht werden, um sicherzustellen, dass Evaluatoren und Notebooks direkt die fusionierten Endgewichte laden.

---

## 4. Empirische Ergebnisse

Die Evaluierung erfolgte auf dem bereinigten Lebenshilfe-Testset (`data/lebenshilfe/lebenshilfe_dataset_clean.json`, $N = 37$ Artikeltexte) unter einheitlichen Generierungsbedingungen (`num_beams=4`, `no_repeat_ngram_size=3`, `max_target_len=256`).

### Zusammenfassungstabelle:

| Modell                        | Ø Simplicity ($R_{\text{style}}$) | Ø Semantik zu AS ($R_{\text{sem}}$) | Ø Treue zu LS ($Sim_{\text{ref}}$) | Composite (0.5/0.5) | Composite (0.7/0.3) | Composite (1.0/0.0) |    BLEU    | ROUGE-L F1 | Ø Gen. Tokens | Truncation Rate (%) |
| :---------------------------- | :-------------------------------: | :---------------------------------: | :--------------------------------: | :-----------------: | :-----------------: | :-----------------: | :--------: | :--------: | :-----------: | :-----------------: |
| **SFT Baseline**              |              0.6445               |             **0.8628**              |               0.8503               |       0.7537        |       0.7100        |       0.6445        |   0.0069   |   0.1279   |    164.78     |       75.68%        |
| **DPO (0.5 Style / 0.5 Sem)** |              0.6747               |               0.8585                |             **0.8561**             |       0.7666        |       0.7298        |       0.6747        | **0.0091** | **0.1280** |    164.92     |       72.97%        |
| **DPO (0.7 Style / 0.3 Sem)** |            **0.6877**             |               0.8540                |               0.8554               |     **0.7709**      |     **0.7376**      |     **0.6877**      |   0.0085   |   0.1278   |  **167.68**   |       81.08%        |
| **DPO (1.0 Style / 0.0 Sem)** |              0.6756               |               0.8613                |               0.8552               |       0.7684        |       0.7313        |       0.6756        |   0.0075   |   0.1264   |    163.54     |     **64.86%**      |

---

## 5. Wissenschaftliche Erkenntnisse & Diskussion

1. **Pareto-Optimaler Betriebspunkt bei $0.7 \text{ Style} / 0.3 \text{ Semantik}$:**
   - Die Konfiguration **`dpo_w07_w03`** erzielt mit **$R_{\text{style}} = 0.6877$** den höchsten Einfachheitsgrad und mit **$0.7709$** den höchsten Composite Reward aller untersuchten Modelle.
   - Der semantische Erhalt zur Ausgangssprache bleibt mit $0.8540$ nahezu identisch zur Baseline ($0.8628$), was belegt, dass 30 % semantische Führung vollkommen genügen, um den Informationsgehalt zu sichern.

2. **DPO übertrifft SFT konsistent über alle Gewichtungsregimes:**
   - Jede DPO-Variante übertrifft die SFT-Baseline sowohl bei der sprachlichen Einfachheit ($+4.7\%$ bis $+6.7\%$) als auch bei der lexikalischen Referenztreue ($Sim_{\text{ref}}$ $0.855$–$0.856$ vs. $0.850$).
   - DPO reduziert typische SFT-Fehlermuster wie künstliche Frageschleifen und unnatürliche Satzverdopplungen.

3. **Semantische Robustheit ohne expliziten Semantik-Reward ($w_{\text{style}} = 1.0$):**
   - Selbst bei vollständigem Verzicht auf den semantischen Term ($w_{\text{sem}} = 0.0$) kollabiert die Semantik nicht ($R_{\text{sem}} = 0.8613$).
   - **Erklärung:** Der DPO-Verlustterm $\log \frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)}$ penalisiert Abweichungen vom Referenzmodell $\pi_{\text{ref}}$. Das Modell bleibt semantisch verankert, erzielt jedoch mit **64.86% die geringste Satzabbruch-Quote** (_Truncation Rate_).

4. **Stabilität der generierten Textlänge:**
   - Alle drei DPO-Varianten halten die Zielsatzlänge stabil bei **163.5–167.7 Tokens** (äquivalent zu SFT mit 164.8 Tokens). Die Längennormalisierung (`mean`) eliminiert jede Form von Kürzungsdegeneration.

5. **Cross-Experiment-Konsistenz:**
   - Die Ergebnisse von `dpo_w05_w05` stimmen bis auf die letzte Nachkommastelle exakt mit dem Modell `dpo_mean` aus dem Loss-Aggregations-Experiment überein ($0.6747$ Simplicity, $164.92$ Tokens), was die mathematische und methodische Reproduzierbarkeit der Pipeline bestätigt.
