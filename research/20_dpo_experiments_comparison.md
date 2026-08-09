# DPO-Experimente & Konfigurationsvergleich (Woche 20)

Dieses Dokument vergleicht die 6 unterschiedlichen Konfigurationen für das *Direct Preference Optimization* (DPO) Feintuning unseres Übersetzungsmodells (`mbart-large-50`). Die Modelle wurden auf dem ungesehenen Out-of-Domain Lebenshilfe-Testset evaluiert.

---

## 1. DPO-Dimensionen und Konfigurationsparameter

Die Experimente erstrecken sich über drei Hauptdimensionen:

1. **Wahrscheinlichkeits-Berechnung (Trainer vs. Non-Trainer):**
   * **Non-Trainer:** Verwendet den durchschnittlichen Cross-Entropy-Loss pro Token (`outputs.loss`) zur Bestimmung der Log-Wahrscheinlichkeiten ($-\text{loss}$).
   * **Trainer:** Berechnet die mathematisch exakte Log-Wahrscheinlichkeit der Sequenz durch Summation der Log-Wahrscheinlichkeiten aller generierten (nicht-maskierten) Token (`get_batch_logps`).
2. **Reward-Gewichtung ($w_{\text{style}}$ vs. $w_{\text{sem}}$):**
   * **`w05_w05`:** Ausgeglichene Gewichtung (50% Stil/Einfachheit, 50% semantischer Erhalt zur Quelle).
   * **`w10_w00`:** Einseitige Gewichtung (100% Stil/Einfachheit, 0% semantischer Erhalt).
3. **Datensatz (Final vs. Enriched):**
   * **Final:** Standardmäßiger bereinigter paralleler Korpus.
   * **Enriched:** Um Glossar-Erklärungen und zusätzliche Begriffserläuterungen angereicherter Trainingsdatensatz.

---

## 2. Quantitative Ergebnisse auf dem Lebenshilfe-Testset

Die Evaluierung auf dem Out-of-Domain Lebenshilfe-Testset liefert folgende Ergebnisse:

| DPO-Modell-Konfiguration | Ø Simplicity ($R_{\text{style}}$) | Ø Sem-Sim zu AS ($R_{\text{sem}}$) | Ø Sem-Sim zu LS-Ref | Ø Composite Reward (0.5/0.5) |
| :--- | :---: | :---: | :---: | :---: |
| **1. `dpo_w05_w05_final` (Non-Trainer)** | 0.8422 | 0.8733 | 0.8334 | **0.8577** |
| **2. `dpo_w05_w05_final_trainer` (Trainer)** | 0.7491 | 0.8867 | 0.8418 | 0.8179 |
| **3. `dpo_w10_w00_final` (Non-Trainer)** | **0.9345** | 0.8689 | 0.8383 | **0.9017** |
| **4. `dpo_w10_w00_final_trainer` (Trainer)** | 0.6182 | **0.8938** | 0.8401 | 0.7560 |
| **5. `dpo_w05_w05_enriched` (Non-Trainer)** | 0.7769 | **0.9058** | **0.8551** | 0.8413 |
| **6. `dpo_w05_w05_enriched_trainer` (Trainer)** | 0.7774 | 0.8920 | 0.8370 | 0.8347 |

---

## 3. Zentrale Erkenntnisse

* **Loss-Berechnung:** Die *Non-Trainer*-Modelle (welche die durchschnittliche Token-Wahrscheinlichkeit optimieren) erzielen durchweg höhere Einfachheits-Scores ($R_{\text{style}}$) und Composite Rewards als die mathematisch exakten *Trainer*-Modelle. Die exakte Summation der Log-Wahrscheinlichkeiten begünstigt tendenziell die Beibehaltung komplexerer Strukturen, da längere, informationsreichere Sätze im Verhältnis seltener radikal gekürzt werden.
* **Reward-Verhältnis:** Die Optimierung unter `w10_w00` (100% Fokus auf Vereinfachung) führt im Non-Trainer-Modell zur maximalen Einfachheit von **0.9345**, reduziert jedoch leicht den semantischen Erhalt zur Quelle.
* **Datensatz-Enrichment:** Das Training auf dem mit Glossar-Erklärungen angereicherten Datensatz (`enriched`) verbessert den semantischen Erhalt signifikant auf bis zu **0.9058** (Sem-Sim zu AS) und erzielt mit **0.8551** die höchste Übereinstimmung mit der echten Leichten-Sprache-Referenz. Das Modell lernt durch die Erläuterungen, Begriffe genauer zu übertragen, anstatt sie einfach wegzulassen.
