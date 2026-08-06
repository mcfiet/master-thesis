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

### Analyse der Lebenshilfe-KDE-Plots

![KDE-Plots Vergleich](/Users/fietescheel/.gemini/antigravity-cli/brain/051373e6-b1cf-40ad-850e-a318e163ff7e/mixup_distribution_comparison_plot.png)

Bei der Evaluation der vier Modelle auf dem ungesehenen Lebenshilfe-Testset fallen folgende strukturelle Unterschiede auf:

#### 1. Hohe Konfidenz bei Alltagssprache (AS)
* **Beobachtung:** Für reine AS-Absätze (blaue Kurve) liefern die Modelle (insb. Variante A, C und D) extrem scharfe, hohe Peaks sehr nahe bei $0.0$. Das bedeutet, das Modell ist sich bei Alltagssprache sehr sicher.
* **Interpretation:** Alltagssprache zeichnet sich durch das Vorhandensein expliziter Komplexitätsmarker aus (z. B. lange Komposita, komplexe Nebensätze, Passivkonstruktionen, spezifisches Fachvokabular). Diese syntaktischen und semantischen Merkmale fungieren als starke "Smoking Guns". Das Modell kann sie leicht identifizieren und schließt mit hoher Wahrscheinlichkeit auf $\lambda \approx 0.0$ (reine AS).

#### 2. Höhere Varianz / "Spielraum" bei Einfacher Sprache (LS)
* **Beobachtung:** Für reine LS-Absätze (grüne Kurve) sind die Dichtekurven deutlich flacher und breiter gestreut. Es gibt hier also mehr Spielraum und weniger absolute Sicherheit.
* **Interpretation:** Einfache Sprache definiert sich primär durch die *Abwesenheit* von Komplexitätsmerkmalen, nutzt aber denselben grundlegenden Wortschatz wie die Alltagssprache. Da dem Modell eindeutige positive Indikatoren für Komplexität fehlen, ist das Signal "weicher". Das Modell muss auf die Abwesenheit komplexer Strukturen schließen, was zu einer breiteren Verteilung der vorhergesagten $\lambda$-Werte führt.

#### 3. Leistung der Modellvarianten im Vergleich
* **Variante A (Statisch) & Variante C (Hybrid):** Zeigen eine gute und saubere Trennung. Der AS-Peak ist sehr scharf bei $\sim 0.08$.
* **Variante B (Dynamisch):** Zeigt die schlechteste Leistung. Der AS-Peak verschiebt sich nach rechts auf $\approx 0.28$ und überlappt stark mit einem Nebenpeak der LS-Verteilung. Das Modell konnte durch das ständige Mischen während des Trainings keine klaren, stabilen Klassengrenzen lernen.
* **Variante D (Hybrid + Cyclic):** Liefert die beste LS-Trennung. Die grüne Kurve ist am weitesten nach rechts verschoben (Peak bei $\approx 0.85$), was zeigt, dass das Modell Einfache Sprache am saubersten und stärksten von der Alltagssprache abgrenzen kann. Die zyklischen Warm-Restarts haben dem Modell geholfen, eine robustere Repräsentation für LS zu lernen.

---

### Zukünftige Optimierungsschritte (Backlog)
* **Lebenshilfe Evaluation:** Testen des trainierten Regressors auf dem externen Lebenshilfe-Testset (`data/lebenshilfe/lebenshilfe_dataset_no_paragraphs.json`), um zu sehen, ob das Modell auf reiner Einfacher Sprache ($\lambda \approx 1.0$) und Alltagssprache ($\lambda \approx 0.0$) verlässliche Ergebnisse liefert.
