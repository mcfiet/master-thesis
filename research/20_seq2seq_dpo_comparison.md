# Vergleich des DPO-Trainings (Commit a968301 vs. Aktuell)

Dieses Dokument dokumentiert den Vergleich des Jupyter-Notebooks [`1_train_seq2seq_dpo.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/translation/1_train_seq2seq_dpo.ipynb) zwischen dem Commit `a968301` und dem aktuellen lokalen Stand.

---

## 1. Übersicht der Konfigurationsunterschiede

| Parameter / Feature | Commit `a968301` | Aktueller Stand | Zweck / Auswirkung |
| :--- | :--- | :--- | :--- |
| **Sprachmodell** | `google/mt5-small` (~300M Parameter) | `facebook/mbart-large-50` (~611M Parameter) | Höhere Modellkapazität für komplexe Übersetzungen in Leichte Sprache. |
| **SFT Epochen** | 10 | 20 | Längere Optimierungsphase für das größere Modell. |
| **Early Stopping Patience** | 2 | 5 | Verhindert verfrühten Abbruch bei langsamerer Konvergenz. |
| **Lernrate (LR)** | `5e-5` | `1e-5` | Stabilisiert das Feintuning des größeren MBart-Modells. |
| **Warmup Steps** | 50 | 150 | Sanfterer Einstieg in die Gradienten-Updates. |
| **Gradient Accumulation** | Keine (`accumulation_steps=1`) | `accumulation_steps=4` | Erhöht die effektive Batchgröße auf 16 zur Trainingsstabilisierung bei limitiertem VRAM. |
| **Mixed Precision (Validation)**| FP32 | BF16 (autocast) | Beschleunigt Validierung und spart Grafikkartenspeicher. |
| **Memory Cleanup vor DPO** | Kein explizites Cleanup | `del optimizer`, `gc.collect()`, `torch.cuda.empty_cache()` | Verhindert CUDA Out-of-Memory (OOM) beim Klonen des Referenzmodells. |
| **DPO-Trainingsdaten** | Subset von 300 Samples | Kompletter Trainingsdatensatz | Robustere DPO-Ausrichtung durch vielfältigere Präferenzpaare. |
| **DPO Batch Size** | 2 | 4 | Stabilere DPO-Gradientenschritte. |
| **Evaluierungs-Batchgröße** | 4 | 16 | Beschleunigt die Übersetzung des Lebenshilfe-Datensatzes. |

---

## 2. Quantitative Ergebnisse im Vergleich

### Supervised Fine-Tuning (SFT)
*   **`mt5-small` (Commit `a968301`):** Konvergiert sehr langsam. Bester Validierungsloss in Epoche 10: **`2.3775`** (Train Loss: `2.8784`).
*   **`mbart-large-50` (Aktuell):** Startet bereits mit besserem Validierungsloss (`2.3914`) und erzielt in Epoche 16 das beste Ergebnis mit **`1.2259`** (Train Loss: `0.5787`).

### DPO Trainingsverlauf
*   **Commit `a968301`:** Brach in Epoche 2 bei 81% ab.
    *   *Epoche 1:* Ø DPO Loss: `0.6931` | Ø Style Reward: `0.9497` | Ø Semantic Reward: `0.8415`
*   **Aktuell:** Erfolgreich über 2 Epochen trainiert.
    *   *Epoche 2:* Ø DPO Loss: `0.6921` | Ø Style Reward: `0.5093` | Ø Semantic Reward: `0.7101`

### Evaluierung auf dem Lebenshilfe-Testset

| Metrik | Commit `a968301` (mt5-small) | Aktueller Stand (mbart-large-50) |
| :--- | :---: | :---: |
| **MixUp-Einfachheits-Score ($R_{\text{style}}$)** | **0.8105 ± 0.0955** | 0.5016 ± 0.0225 |
| **SBERT-Ähnlichkeit zur AS-Quelle ($R_{\text{sem}}$)** | 0.8773 ± 0.0584 | **0.8947 ± 0.0552** |
| **SBERT-Ähnlichkeit zur LS-Referenz** | 0.8428 ± 0.0719 | 0.8425 ± 0.0810 |
| **Composite Reward** | **0.8439 ± 0.0565** | 0.6981 ± 0.0313 |

> [!NOTE]
> Obwohl das kleinere `mt5-small` formal einen höheren Einfachheits- und Composite-Reward erzielt, neigen kleinere Modelle bei diesen Metriken zur Ausnutzung von Heuristiken des Klassifikators (z. B. extrem kurze Sätze oder repetitive einfache Phrasen), was sich negativ auf die inhaltliche Qualität auswirkt.

---

## 3. Qualitative Stichprobenanalyse

### Beispiel 1: Übersetzung einer Presseinformation

*   **Ausgangssprache (AS-Quelle):**
    > „Inklusion im Fokus“: Aktionstag an der CAU bringt alle zusammen.
*   **`mt5-small` (Commit-Version):**
    > *„Manche Menschen mit Behinderungen sind sehr vielfältig. Das ist Leben - einzigartig Vielleicht gibt es viele Dinge. In diesem Tag hat die Landesregierung eine wichtige Veranstaltung. Und das heißt: „Inklusion im Fokus“. Der 3. Dezember ist der 2. Dezember. Die Nationalsozialisten haben viele Informationen zum Aktionstag an der CAU und StK. [...]“*
    >
    > **Kritik:** Starke Halluzinationen (z. B. fälschliche Erwähnung der Nationalsozialisten, Datumsfehler *"3. Dezember ist der 2. Dezember"*), grammatikalisch wirr und inhaltlich hochgradig verfälscht. SBERT-Similarity zur echten Referenz: **0.579**.
*   **`mbart-large-50` (Aktuelle Version):**
    > *„Presse-Mitteilung von der Christian-Albrechts-Universität zu Kiel 3.12.2024 xxx/2024 Aktionstag an der CAU: Menschen mit Behinderungen treffen sich bei Film-Vortrag Daniel Günther eröffnet Veranstaltung mit Film „Das ist Leben - einzigartig vielfältig“ Am 3. Dezember ist der Internationale Tag der Menschen ohne Behinderungen. [...] Die Politiker haben darüber gesprochen: • Welche Veränderungen wir brauchen, damit alle Menschen gleichberechtigt leben können. • Und wie wir diese Veränderungen erreichen können. [...]“*
    >
    > **Kritik:** Fließender Text, präzise Formulierung, korrekter Informationsgehalt und hervorragende Strukturierung durch Bulletpoints in Leichter Sprache. SBERT-Similarity zur Quelle: **0.965**.
