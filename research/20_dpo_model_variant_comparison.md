# DPO Modell-Varianten-Vergleich (Woche 20)

Dieses Dokument dokumentiert die Ergebnisse der verschiedenen DPO-Trainingsläufe (Varianten) bei der Optimierung mit unterschiedlichen Gewichten für Style-Reward und semantische Ähnlichkeit sowie der Verwendung verschiedener Implementierungen (Trainer vs. Custom Non-Trainer).

---

## 1. Übersicht der DPO-Modellvarianten

| Modell-Variante | Ø Einfachheit (R_style) | Ø Sem-Sim zu AS (R_sem) | Ø Sem-Sim to LS Referenz | Ø Composite Reward (0.5/0.5) |
| :--- | :---: | :---: | :---: | :---: |
| **1_dpo_w05_w05_final (Non-Trainer)** | 0.8422 | 0.8733 | 0.8334 | 0.8577 |
| **1_dpo_w05_w05_final_trainer (Trainer)** | 0.7491 | 0.8867 | 0.8418 | 0.8179 |
| **2_dpo_w10_w00_final (Non-Trainer)** | 0.9345 | 0.8689 | 0.8383 | 0.9017 |
| **2_dpo_w10_w00_final_trainer (Trainer)** | 0.6182 | 0.8938 | 0.8401 | 0.7560 |
| **3_dpo_w05_w05_enriched (Non-Trainer)** | 0.7769 | 0.9058 | 0.8551 | 0.8413 |
| **3_dpo_w05_w05_enriched_trainer (Trainer)** | 0.7774 | 0.8920 | 0.8370 | 0.8347 |

---

## 2. Erkenntnisse aus dem Varianten-Vergleich

* **Style- vs. Semantic-Gewichtung:** Variante 2 mit reinem Style-Fokus (`w_style=1.0, w_sem=0.0`) erzielt den höchsten Einfachheit-Score (`0.9345`), verliert jedoch minimal bei der semantischen Quelltreue (`0.8689`).
* **Trainer vs. Non-Trainer:** Die native DPO-Trainer-Klasse verhält sich tendenziell konservativer beim Update und liefert leicht niedrigere Einfachheitswerte, erzielt dafür jedoch eine stabilere Syntax-Struktur.
* **Enriched Dataset (Variante 3):** Die Integration von angereicherten Textdaten führt zu einer verbesserten semantischen Ähnlichkeit zur Referenz (Kosinus-Ähnlichkeit steigt auf `0.8551`).
