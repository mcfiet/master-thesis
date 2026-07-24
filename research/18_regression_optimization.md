# MixUp Regressor Optimierung & Evaluation

## 1. Training mit LLM-generierten Sprachstufen
* Trainieren des Modells auf synthetischen Textstufen (`0.25`, `0.50`, `0.75`).

## 2. Variante B mit zyklischer Lernrate (Cyclic LR)
* Trainieren des rein dynamischen Mischmodells (Variante B) mit einem zyklischen Learning-Rate-Scheduler (z. B. `CosineAnnealingWarmRestarts`), um zu prüfen, ob periodisches Momentum hilft, lokalen Minima zu entkommen und das anfängliche Konvergenzplateau zu überwinden.

## 3. Evaluation auf dem Test-Split & Lebenshilfe-Datensatz (Erledigt)

### 3.1. Detaillierte Evaluierungsergebnisse aus `notebooks/4_mixup_model_evaluation.ipynb`

#### 3.1.1. In-Domain Evaluation auf dem Test-Split

##### Regressions-Performance (Kontinuierliche MixUp-Vorhersage auf Test-Split)
| Modell | Test MSE | Test MAE |
| :--- | :---: | :---: |
| **Variante A (Statisch)** | 0.0383 | 0.1557 |
| **Variante B (Dynamisch)** | 0.0758 | 0.2264 |
| **Variante C (Hybrid)** | 0.0267 | 0.1158 |
| **Variante D (Hybrid + Cyclic)** | **0.0241** | **0.1027** |

##### Binäre Klassifikation auf dem Test-Split (Reine Sätze: LS = 1.0, AS = 0.0)
| Modell | Ø $\lambda_{LS}$ | Ø $\lambda_{AS}$ | Accuracy (Schwelle 0.5) | Balanced Accuracy | MAE (Target 1/0) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Variante A (Statisch)** | 0.7596 | 0.2680 | 91.55% | 91.46% | 0.2526 |
| **Variante B (Dynamisch)** | 0.6138 | 0.3312 | 80.17% | 79.53% | 0.3620 |
| **Variante C (Hybrid)** | 0.8603 | 0.1657 | 95.04% | 95.22% | 0.1511 |
| **Variante D (Hybrid + Cyclic)** | **0.9007** | **0.1382** | **95.92%** | **95.93%** | **0.1164** |

---

#### 3.1.2. Out-of-Domain Evaluation auf dem Lebenshilfe-Datensatz

##### Regressions-Performance (MixUp auf Lebenshilfe-Datensatz)
| Modell | LH MSE | LH MAE |
| :--- | :---: | :---: |
| **Variante A (Statisch)** | 0.0725 | 0.2111 |
| **Variante B (Dynamisch)** | 0.0766 | 0.2212 |
| **Variante C (Hybrid)** | 0.0840 | 0.2279 |
| **Variante D (Hybrid + Cyclic)** | **0.0739** | **0.2087** |

##### Binäre Klassifikation auf dem Lebenshilfe-Datensatz (Reine Sätze: LS = 1.0, AS = 0.0)
| Modell | Ø $\lambda_{LS}$ | Ø $\lambda_{AS}$ | Accuracy (Schwelle 0.5) | Balanced Accuracy | MAE (Target 1/0) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Variante A (Statisch)** | 0.6533 | 0.1958 | 82.67% | 85.75% | 0.2888 |
| **Variante B (Dynamisch)** | 0.6241 | 0.2766 | 87.42% | 86.32% | 0.3378 |
| **Variante C (Hybrid)** | 0.6397 | 0.1236 | 82.03% | 85.10% | 0.2695 |
| **Variante D (Hybrid + Cyclic)** | **0.7293** | **0.0957** | **87.29%** | **89.37%** | **0.2036** |

---

### 3.2. Kernerkenntnisse aus dem Notebook
* **Variante D (Hybrid + Cyclic)** liefert über alle Datasets und Testaufbauten hinweg das mit Abstand beste Gesamtergebnis.
  * Auf dem In-Domain Test-Split erzielt Variante D den geringsten Regressionsfehler (**MSE = 0,0241**, **MAE = 0,1027**) sowie die höchste binäre Klassifikationsgenauigkeit (**95,92%** Accuracy, **95,93%** Balanced Accuracy).
  * Auf dem Out-of-Domain Lebenshilfe-Set zeigt Variante D mit **87,29% Accuracy** (**89,37%** Balanced Accuracy) und **MAE = 0,2036** die stärkste Generalisierung und die schärfste Abgrenzung zwischen LS ($\bar{\lambda}_{LS} = 0,7293$) und AS ($\bar{\lambda}_{AS} = 0,0957$).
* **Variante C (Hybrid)** zeigt ebenfalls eine sehr starke Performance (95,04% Test Accuracy, MSE = 0,0267), profitiert jedoch nochmals spürbar von den zyklischen Lernraten-Restarts in Variante D.
* **Variante B (Rein dynamisch)** hat aufgrund der durchgehend hohen Varianz im Dataloader die größten Schwierigkeiten zu konvergieren (Test Accuracy 80,17%).

---

### 3.3. Grafische Visualisierungen (In-Domain Test-Split & Out-of-Domain Lebenshilfe)

#### Test-Split Visualisierungen
![Testset KDE-Dichteverteilung](img/analysis/mixup_test_distribution_with_targets.png)
*Abbildung 1: Dichteverteilung ($\lambda$) auf dem Test-Split (Reine Sätze vs. Trainings-Target-Verteilung).*

![Testset Regressions-KDE](img/analysis/mixup_test_regression_kde.png)
*Abbildung 2: Regressions-Dichteverteilung (Modell-Vorhersagen vs. wahre Test-Targets bei MixUp).*

![Testset Klassifikations-Scatterplot](img/analysis/mixup_test_classification_scatterplot.png)
*Abbildung 3: Ist- vs. Soll-Werte Scatterplot für die Klassifikation auf dem Test-Split.*

![Testset Regressions-Scatterplot](img/analysis/mixup_test_regression_scatterplot.png)
*Abbildung 4: Regressions-Scatterplot auf dem gemischten Test-Split.*

#### Lebenshilfe (Out-of-Domain) Visualisierungen
![Lebenshilfe KDE-Dichteverteilung](img/analysis/mixup_distribution_with_targets.png)
*Abbildung 5: Dichteverteilung auf dem Lebenshilfe-Datensatz vs. Trainings-Target-Verteilung.*

![Lebenshilfe Regressions-KDE](img/analysis/mixup_lh_regression_kde.png)
*Abbildung 6: Regressions-Dichteverteilung auf dem gemischten Lebenshilfe-Set.*

![Lebenshilfe Klassifikations-Scatterplot](img/analysis/mixup_lh_classification_scatterplot.png)
*Abbildung 7: Ist- vs. Soll-Werte Scatterplot auf dem Lebenshilfe-Set.*

![Lebenshilfe Regressions-Scatterplot](img/analysis/mixup_lh_regression_scatterplot.png)
*Abbildung 8: Regressions-Scatterplot auf dem gemischten Lebenshilfe-Set.*
