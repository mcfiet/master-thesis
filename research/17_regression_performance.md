## MixUp Regression Training verbessern

Hier dokumentieren wir die Erkenntnisse und den theoretischen Hintergrund zu den vier verschiedenen Implementierungen des MixUp-Regressors:

### Die vier MixUp-Varianten im Vergleich

#### 1. Variante A: Statische Vormischung in `__init__` (`3_mixup_dataloader_test.ipynb`)
* **Konzept:** Alle Satzmischungen und das Shuffling werden einmalig bei der Initialisierung des Datasets fest generiert.
* **Vorteil (Stabile Konvergenz):** Sehr stabiles anfängliches Lernen, da das Modell in jeder Epoche exakt dieselben Datenpunkte sieht (geringe Varianz).
* **Nachteil (Overfitting-Gefahr):** Das Modell sieht keine neuen Kombinationen der Sätze über die Epochen hinweg. Die Anzahl der Muster ist starr auf `mixtures_per_pair` begrenzt, was die Effizienz der Augmentierung einschränkt.

#### 2. Variante B: Rein dynamisches On-the-Fly Shuffling in `__getitem__` (`3_mixup_dataloader_test_getitem.ipynb`)
* **Konzept:** Jedes Mal, wenn ein Batch geladen wird (`__getitem__`), werden die Sätze neu und zufällig zusammengestellt und geshuffelt.
* **Vorteil (Hohe Augmentierung):** Maximale Varianz. Das Modell sieht in jeder Epoche neue Satzkombinationen, was Overfitting stark entgegenwirkt und die Generalisierung verbessert.
* **Nachteil (Schwierige Konvergenz):** Schlechte oder sehr langsame Konvergenz zu Beginn des Trainings, da die ständige Änderung der Daten (hohe Varianz) die Gradienten unruhig macht und das Modell Schwierigkeiten hat, stabile Repräsentationen zu lernen.

#### 3. Variante C: Hybrid-Lösung (`3_mixup_dataloader_test_hybrid.ipynb`)
* **Konzept:** Kombination aus beiden Welten. Das Dataset hält sowohl die vormischten statischen Samples als auch die Roh-Sätze.
  * In Epoche 0 ist die Wahrscheinlichkeit für dynamisches Mischen $p_{dynamic} = 0.0$ (rein statisch).
  * Die Wahrscheinlichkeit steigt linear mit den Epochen: $p_{dynamic} = \frac{\text{aktuelle\_epoche}}{\text{gesamt\_epochen} - 1}$.
  * In der letzten Epoche ist $p_{dynamic} = 1.0$ (rein dynamisch).
* **Vorteil (Beste Balance):** Erlaubt dem Modell eine stabile und schnelle Konvergenz in der frühen Phase (durch die statischen Samples) und führt in der Spätphase eine hohe Varianz ein (durch das dynamische Mischen), um Overfitting zu verhindern.

#### 4. Variante D: Hybrid-Lösung + Zyklischer LR-Scheduler (`3_mixup_dataloader_test_hybrid_cyclic.ipynb`)
* **Konzept:** Basiert auf der Hybrid-Lösung (Variante C), führt aber zusätzlich einen zyklischen Learning-Rate-Scheduler (`CosineAnnealingWarmRestarts`) ein.
* **Vorteil (Erleichterter Phasenübergang & Minima-Flucht):** 
  * Der Übergang von statischen zu dynamischen Daten verändert die Stabilität der Gradienten. Die periodische Anpassung der Lernrate hilft dem Optimizer, sich auf das Rauschen einzustellen.
  * Durch das periodische Anheben der Lernrate ("Warm Restarts") erhält das Modell genügend Momentum, um aus suboptimalen lokalen Minima oder Sattelpunkten auszubrechen, in die es während der Datenverschiebung geraten sein könnte.

---

### Zukünftige Optimierungsschritte (Backlog)
* **Lebenshilfe Evaluation:** Testen des trainierten Regressors auf dem externen Lebenshilfe-Testset (`results/lebenshilfe_dataset_no_paragraphs.json`), um zu sehen, ob das Modell auf reiner Einfacher Sprache ($\lambda \approx 1.0$) und Alltagssprache ($\lambda \approx 0.0$) verlässliche Ergebnisse liefert.
