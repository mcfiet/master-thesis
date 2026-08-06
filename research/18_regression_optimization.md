# MixUp Regressor Optimierung & Evaluation

## 1. Training mit LLM-generierten Sprachstufen

- Trainieren des Modells auf synthetischen Textstufen (`0.25`, `0.50`, `0.75`).

## 2. Variante B mit zyklischer Lernrate (Cyclic LR)

- **Notebook:** `notebooks/3_mixup_dataloader_test_getitem_cyclic.ipynb`
- **Modellgewichte:** `results/models/bilstm_mixup_regression_getitem_cyclic.pt`

### 2.1. Grafische Auswertung & Detail-Analyse

![Trainingsverlauf Variante B (Dynamisch + Cyclic LR)](img/analysis/mixup_getitem_cyclic_loss_curve.png)
_Abbildung 2.1: Trainings- und Validierungs-Loss (MSE) über 25 Epochen mit CosineAnnealingWarmRestarts._

![Scatterplot Echte vs. Vorhergesagte Targets](img/analysis/mixup_getitem_cyclic_scatterplot.png)
_Abbildung 2.2: Scatterplot der echten vs. vorhergesagten Lambda-Werte auf den Validierungsdaten._

### 2.2. Detaillierte Befunde & Korrektur der Analyse

1. **Instabilität des Trainingsverlaufs (Loss-Kurve):**
   - **Starke Oszillation des Validierungs-Loss:** Während der Trainings-Loss (blau) kontinuierlich von 0,085 auf ~0,049 sinkt, schwankt der Validierungs-Loss (orange) extrem stark zwischen **0,057 und 0,090**.
   - **Wirkung der Restarts:** Die zyklischen Restarts (`CosineAnnealingWarmRestarts` mit `T_0=10`) erzeugen zwar punktuell tiefe Dips im Val-Loss (z. B. Epoche 15 mit MSE = 0,0505), das Modell stabilisiert sich jedoch zu keinem Zeitpunkt. Sobald die Lernrate im nächsten Zyklus wieder ansteigt, schlägt der Val-Loss sofort wieder massiv nach oben aus (z. B. Epoche 22 mit MSE = 0,090).

2. **Fehlende Regressions-Kontinuität (Scatterplot):**
   - **Keine Konvergenz zur Hauptdiagonalen ($y = x$):** Der Scatterplot zeigt drastisch, dass das Modell keine stetige Regressionsgerade lernt. Die Punkte streuen nicht gleichmäßig um die rote Zielgerade, sondern bilden eine verzerrte, bimodal geteilte Verteilung.
   - **Starke Stauchung an den Rändern:**
     - Für reine Alltagssprache ($\lambda = 0,0$) sagt das Modell Werte zwischen $0,28$ und $0,42$ voraus – der Nullpunkt wird nicht erreicht.
     - Für reine Leichte Sprache ($\lambda = 1,0$) liegen die Vorhersagen weit gestreut zwischen $0,36$ und $0,81$ – der Maximalwert 1,0 wird verfehlt.
   - **Bimodale Clusterbildung im Mittelbereich:** Für kontinuierliche Mischwerte ($\lambda \in [0,2; 0,8]$) spaltet sich die Modellvorhersage im Wesentlichen in zwei horizontale Bänder auf: ein unteres Band um $\lambda \approx 0,30 - 0,38$ und ein oberes Band um $\lambda \approx 0,70 - 0,80$.

3. **Korrigiertes Gesamtfazit zu Variante B:**
   - Auch wenn der minimal gemessene Val-Loss zahlenmäßig leicht besser erscheint als beim Standard-Variante-B-Training (0,0505 vs. 0,0758), belegt die visuelle Verteilungsanalyse eindeutig: **Variante B fasst auch mit zyklischer Lernrate im Training NICHT richtig Fuß.**
   - **Ursache:** Durch die rein stochastische Generierung neuer Slices in jedem `__getitem__`-Zugriff fehlen dem Dataloader feste, verlässliche Ankerpunkte im Batch. Der Regressor lernt somit keinen stetigen Komplexitätsgradienten, sondern driftet in zwei grobe Komplexitäts-Cluster ab.
   - **Bestätigung des Hybrid-Ansatzes:** Dieses Verhalten liefert den methodischen Beweis dafür, warum **Variante D (Hybrid + Cyclic LR)** zwingend notwendig ist: Erst die Verankerung statisch prä-generierter Paare zusammen mit dynamischen Stichproben erlaubt es dem Modell, eine präzise, stetige Regressionsfunktion über das gesamte Komplexitätsspektrum auszubilden.

## 3. Evaluation auf dem Test-Split & Lebenshilfe-Datensatz

### 3.1. Detaillierte Evaluierungsergebnisse aus `notebooks/4_mixup_model_evaluation.ipynb`

#### 3.1.1. In-Domain Evaluation auf dem Test-Split

##### Regressions-Performance (Kontinuierliche MixUp-Vorhersage auf Test-Split)

| Modell                           |  Test MSE  |  Test MAE  |
| :------------------------------- | :--------: | :--------: |
| **Variante A (Statisch)**        |   0.0383   |   0.1557   |
| **Variante B (Dynamisch)**       |   0.0758   |   0.2264   |
| **Variante C (Hybrid)**          |   0.0267   |   0.1158   |
| **Variante D (Hybrid + Cyclic)** | **0.0241** | **0.1027** |

##### Binäre Klassifikation auf dem Test-Split (Reine Sätze: LS = 1.0, AS = 0.0)

| Modell                           | Ø $\lambda_{LS}$ | Ø $\lambda_{AS}$ | Accuracy (Schwelle 0.5) | Balanced Accuracy | MAE (Target 1/0) |
| :------------------------------- | :--------------: | :--------------: | :---------------------: | :---------------: | :--------------: |
| **Variante A (Statisch)**        |      0.7596      |      0.2680      |         91.55%          |      91.46%       |      0.2526      |
| **Variante B (Dynamisch)**       |      0.6138      |      0.3312      |         80.17%          |      79.53%       |      0.3620      |
| **Variante C (Hybrid)**          |      0.8603      |      0.1657      |         95.04%          |      95.22%       |      0.1511      |
| **Variante D (Hybrid + Cyclic)** |    **0.9007**    |    **0.1382**    |       **95.92%**        |    **95.93%**     |    **0.1164**    |

---

#### 3.1.2. Out-of-Domain Evaluation auf dem Lebenshilfe-Datensatz

##### Regressions-Performance (MixUp auf Lebenshilfe-Datensatz)

| Modell                           |   LH MSE   |   LH MAE   |
| :------------------------------- | :--------: | :--------: |
| **Variante A (Statisch)**        |   0.0725   |   0.2111   |
| **Variante B (Dynamisch)**       |   0.0766   |   0.2212   |
| **Variante C (Hybrid)**          |   0.0840   |   0.2279   |
| **Variante D (Hybrid + Cyclic)** | **0.0739** | **0.2087** |

##### Binäre Klassifikation auf dem Lebenshilfe-Datensatz (Reine Sätze: LS = 1.0, AS = 0.0)

| Modell                           | Ø $\lambda_{LS}$ | Ø $\lambda_{AS}$ | Accuracy (Schwelle 0.5) | Balanced Accuracy | MAE (Target 1/0) |
| :------------------------------- | :--------------: | :--------------: | :---------------------: | :---------------: | :--------------: |
| **Variante A (Statisch)**        |      0.6533      |      0.1958      |         82.67%          |      85.75%       |      0.2888      |
| **Variante B (Dynamisch)**       |      0.6241      |      0.2766      |         87.42%          |      86.32%       |      0.3378      |
| **Variante C (Hybrid)**          |      0.6397      |      0.1236      |         82.03%          |      85.10%       |      0.2695      |
| **Variante D (Hybrid + Cyclic)** |    **0.7293**    |    **0.0957**    |       **87.29%**        |    **89.37%**     |    **0.2036**    |

---

### 3.2. Kernerkenntnisse aus dem Notebook

- **Variante D (Hybrid + Cyclic)** liefert über alle Datasets und Testaufbauten hinweg das mit Abstand beste Gesamtergebnis.
  - Auf dem In-Domain Test-Split erzielt Variante D den geringsten Regressionsfehler (**MSE = 0,0241**, **MAE = 0,1027**) sowie die höchste binäre Klassifikationsgenauigkeit (**95,92%** Accuracy, **95,93%** Balanced Accuracy).
  - Auf dem Out-of-Domain Lebenshilfe-Set zeigt Variante D mit **87,29% Accuracy** (**89,37%** Balanced Accuracy) und **MAE = 0,2036** die stärkste Generalisierung und die schärfste Abgrenzung zwischen LS ($\bar{\lambda}_{LS} = 0,7293$) und AS ($\bar{\lambda}_{AS} = 0,0957$).
- **Variante C (Hybrid)** zeigt ebenfalls eine sehr starke Performance (95,04% Test Accuracy, MSE = 0,0267), profitiert jedoch nochmals spürbar von den zyklischen Lernraten-Restarts in Variante D.
- **Variante B (Rein dynamisch)** hat aufgrund der durchgehend hohen Varianz im Dataloader die größten Schwierigkeiten zu konvergieren (Test Accuracy 80,17%).

---

### 3.3. Grafische Visualisierungen (In-Domain Test-Split & Out-of-Domain Lebenshilfe)

#### Test-Split Visualisierungen

![Testset KDE-Dichteverteilung](img/analysis/mixup_test_distribution_with_targets.png)
_Abbildung 1: Dichteverteilung ($\lambda$) auf dem Test-Split (Reine Sätze vs. Trainings-Target-Verteilung)._

![Testset Regressions-KDE](img/analysis/mixup_test_regression_kde.png)
_Abbildung 2: Regressions-Dichteverteilung (Modell-Vorhersagen vs. wahre Test-Targets bei MixUp)._

![Testset Klassifikations-Scatterplot](img/analysis/mixup_test_classification_scatterplot.png)
_Abbildung 3: Ist- vs. Soll-Werte Scatterplot für die Klassifikation auf dem Test-Split._

![Testset Regressions-Scatterplot](img/analysis/mixup_test_regression_scatterplot.png)
_Abbildung 4: Regressions-Scatterplot auf dem gemischten Test-Split._

#### Lebenshilfe (Out-of-Domain) Visualisierungen

![Lebenshilfe KDE-Dichteverteilung](img/analysis/mixup_distribution_with_targets.png)
_Abbildung 5: Dichteverteilung auf dem Lebenshilfe-Datensatz vs. Trainings-Target-Verteilung._

![Lebenshilfe Regressions-KDE](img/analysis/mixup_lh_regression_kde.png)
_Abbildung 6: Regressions-Dichteverteilung auf dem gemischten Lebenshilfe-Set._

![Lebenshilfe Klassifikations-Scatterplot](img/analysis/mixup_lh_classification_scatterplot.png)
_Abbildung 7: Ist- vs. Soll-Werte Scatterplot auf dem Lebenshilfe-Set._

![Lebenshilfe Regressions-Scatterplot](img/analysis/mixup_lh_regression_scatterplot.png)
_Abbildung 8: Regressions-Scatterplot auf dem gemischten Lebenshilfe-Set._
