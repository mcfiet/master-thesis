# Einfachheits-Metriken & Bias-Analyse (Woche 20)

Dieses Dokument dokumentiert die Untersuchungen zur Stabilität unserer automatischen Evaluierungsmetriken, insbesondere bezüglich systematischer Längen- und Boilerplate-Verzerrungen (Biases) sowie den systematischen Vergleich der beiden Regressions-Ansätze.

---

## 1. Untersuchung des Längen- und Boilerplate-Bias

Es wurde untersucht, ob die Einfachheits-Klassifikatoren (unsere Regressoren für den Style-Reward) einen systematischen Längen-Bias aufweisen (d. h. kürzere Sätze unabhängig vom Inhalt fälschlicherweise als "einfacher" bewerten):

* **Methode:** Synthetische Generierung von Sätzen mit identischem Komplexitätsgrad, aber unterschiedlichen Wort- und Zeichenlängen durch Auffüllen mit bedeutungslosen Boilerplate-Phrasen.
* **Ergebnis:** Es konnte nachgewiesen werden, dass manche Metrikmodelle empfindlich auf die reine Textlänge reagieren. Diese Verzerrungen wurden quantifiziert, um im DPO-Tuning Fehlsteuerungen (Reward Hacking durch reine Textkürzung) entgegenzuwirken.

---

## 2. Metrische Ähnlichkeitsprüfung und Korrelation

* **Ziel:** Messung der Korrelation zwischen verschiedenen automatischen Lesbarkeitsindizes (Flesch, Wiener Sachtextformel) und unseren tiefen neuronalen Einfachheits-Scores.
* **Ergebnis:** Unsere gelernten Klassifikatoren zeigen eine starke Korrelation mit den linguistisch begründeten Formeln, weisen aber bei modernen Textstrukturen und idiomatischen Vereinfachungen eine deutlich höhere Robustheit auf.

---

## 3. Einfachheits-Metriken: MixUp- vs. Synthetischer Ansatz

Beide Regressionsmodelle (trainiert über den MixUp-Ansatz bzw. den synthetischen Ansatz) wurden kreuzweise auf beiden Datensätzen (LLM-generierte Stufen und Sentence-MixUp) evaluiert, um ihre Generalisierungsfähigkeit und Komplexitätsbestimmung zu validieren.

### 3.1 Evaluierung auf dem LLM-generierten Stufen-Datensatz

| Metrik | MixUp-Ansatz (Variante D) | Synthetischer Ansatz |
| :--- | :---: | :---: |
| **MSE** | 0.1388 | **0.0786** |
| **MAE** | 0.3079 | **0.1816** |
| **Pearson r** | 0.6626 | **0.7412** |
| **Spearman rho** | 0.5885 | **0.7391** |

Das auf dem synthetischen Ansatz trainierte Modell erzielt in seiner eigenen Generierungs-Domäne (den LLM-generierten Stufen) hervorragende Ergebnisse.

### 3.2 Evaluierung auf dem Sentence-MixUp-Datensatz

| Metrik | MixUp-Ansatz (Variante D) | Synthetischer Ansatz |
| :--- | :---: | :---: |
| **MSE** | **0.0892** | 0.1162 |
| **MAE** | **0.2302** | 0.2629 |
| **Pearson r** | **0.6939** | 0.5979 |
| **Spearman rho** | **0.7045** | 0.5917 |

Der MixUp-Ansatz zeigt eine höhere Stabilität und Robustheit bei der Evaluierung auf dem Sentence-MixUp-Datensatz.
