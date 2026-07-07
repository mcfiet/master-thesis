# Regressions-Training & Mix-Up-Evaluierung (Woche 16)

In dieser Woche (Woche 16) wird mit dem Training des Regressionsmodells auf Basis des Mix-Up-Ansatzes begonnen. Zu Beginn wurden die zusammengestellten Absätze aus dem Mix-Up-Dataloader ausgegeben und analysiert, um die Qualität und Struktur der erzeugten Trainingsdaten manuell zu überprüfen.

---

## 1. Analyse der gemischten Absätze (Mix-Up-Dataloader)

Der Mix-Up-Dataloader schneidet unabhängig voneinander contiguous (zusammenhängende) Slices aus den Leichte-Sprache- (LS) und Alltagssprache- (AS) Versionen eines Artikels aus, führt diese zusammen, shuffelt sie und berechnet das Regressionstarget $\lambda$ als Verhältnis der Zeichenlänge des LS-Anteils zur Gesamtlänge.

### Beispiel eines zusammengestellten Mix-Up-Absatzes

Aus dem Notebook [3_mixup_dataloader_test.ipynb](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/3_mixup_dataloader_test.ipynb) wurde das folgende Beispiel generiert und analysiert:

#### LS-Sätze (Extrakt, $n = 2$):

- _Die Beauftragten der Bundes-Regierung für die Belange von Menschen mit Behinderungen haben viele Aufgaben._
- _Was macht der Behindertenbeauftragte der Bundesregierung?_

#### AS-Sätze (Extrakt, $n = 5$):

- _Inhaltsverzeichnis Video: Was macht der Behindertenbeauftragte der Bundesregierung?_
- _Gesetzlicher Auftrag Politische und soziale Rahmenbedingungen mitgestalten Informieren – beraten – Öffentlichkeitsarbeit leisten – Inklusionsgedanken verbreiten Grenzen der Beratung Video: Was macht der Behindertenbeauftragte der Bundesregierung?_
- _Video: Was macht der Behindertenbeauftragte der Bundesregierung?_
- _zum Download: Video: Was macht der Behindertenbeauftragte der Bundesregierung?_
- _(307 MB, 02:39) Gesetzlicher Auftrag Der/Die Behindertenbeauftragte wird vom Bundeskabinett jeweils für die Dauer einer Legislaturperiode bestellt._

#### Zusammengestellter Absatz (gemischt und geshuffelt):

> Die Beauftragten der Bundes-Regierung für die Belange von Menschen mit Behinderungen haben viele Aufgaben. zum Download: Video: Was macht der Behindertenbeauftragte der Bundesregierung? Video: Was macht der Behindertenbeauftragte der Bundesregierung? (307 MB, 02:39) Gesetzlicher Auftrag Der/Die Behindertenbeauftragte wird vom Bundeskabinett jeweils für die Dauer einer Legislaturperiode bestellt. Gesetzlicher Auftrag Politische und soziale Rahmenbedingungen mitgestalten Informieren – beraten – Öffentlichkeitsarbeit leisten – Inklusionsgedanken verbreiten Grenzen der Beratung Video: Was macht der Behindertenbeauftragte der Bundesregierung? Was macht der Behindertenbeauftragte der Bundesregierung? Inhaltsverzeichnis Video: Was macht der Behindertenbeauftragte der Bundesregierung?

#### Visualisierte Satz-Herkunft im Absatz:

- **[LS]** Die Beauftragten der Bundes-Regierung für die Belange von Menschen mit Behinderungen haben viele Aufgaben.
- **[AS]** zum Download: Video: Was macht der Behindertenbeauftragte der Bundesregierung?
- **[AS]** Video: Was macht der Behindertenbeauftragte der Bundesregierung?
- **[AS]** (307 MB, 02:39) Gesetzlicher Auftrag Der/Die Behindertenbeauftragte wird vom Bundeskabinett jeweils für die Dauer einer Legislaturperiode bestellt.
- **[AS]** Gesetzlicher Auftrag Politische und soziale Rahmenbedingungen mitgestalten Informieren – beraten – Öffentlichkeitsarbeit leisten – Inklusionsgedanken verbreiten Grenzen der Beratung Video: Was macht der Behindertenbeauftragte der Bundesregierung?
- **[LS]** Was macht der Behindertenbeauftragte der Bundesregierung?
- **[AS]** Inhaltsverzeichnis Video: Was macht der Behindertenbeauftragte der Bundesregierung?

#### Berechnetes Regressionstarget ($\lambda$):

Das Target berechnet sich auf Basis des Verhältnisses der Zeichenlänge des LS-Anteils zur Gesamtzeichenlänge:
$$\lambda = \frac{\text{Länge}(LS)}{\text{Länge}(LS) + \text{Länge}(AS)} \approx 0.2087$$

---

## 2. Bewertung der Mix-Up-Struktur

Die manuelle Durchsicht zeigt:

1. **Linguistische Kohärenz:** Durch das Mischen geht der logische und thematische Zusammenhang des Absatzes verloren. Da das Modell jedoch darauf trainiert wird, die Komplexität auf Satz- und Stilebene zu bewerten (und nicht die logische Textfortführung), ist diese Zerstörung der Absatzkohärenz vorteilhaft, um Overfitting auf semantische Muster zu vermeiden.
2. **Target-Eigenschaften:** Das berechnete Target von $\approx 0.2087$ spiegelt den hohen Anteil an komplexeren Alltagssprach-Sätzen (5 von 7 Sätzen) im Verhältnis gut wider.

---

---

## 3. Implementierung & Training des Regressionsmodells (BiLSTM)

In Woche 16 wurde das Regressions-Training auf Basis der gemischten Absätze umgesetzt. Dabei gab es eine Entwicklung von einer ersten instabilen Pipeline hin zu einem voll funktionsfähigen, konvergierten Modell.

### 3.1. Versuchsaufbau & Initiale Implementierung (Fehlermodus)

Zunächst wurde ein `BiLSTMRegressor` mit folgenden Parametern aufgesetzt:

- **Embedding-Größe:** 128 (Padding-Index: 0)
- **Hidden-Dimension:** 128 (bidirektional)
- **Dropout:** 0.3
- **Optimierungsfunktion:** AdamW mit $LR = 10^{-3}$
- **Verlustfunktion (Loss):** MSE-Loss (Mean Squared Error)

Bei dieser ersten Implementierung wurde die Mischung der Sätze (Slicing & Shuffling) und die Target-Berechnung **on-the-fly** während des Aufrufs von `__getitem__` im DataLoader durchgeführt.

#### Problem & Fehlerbild:

Nach 20 Epochen stoppte das Training mit einem scheinbar soliden Validation MSE von `0.0655` und einem MAE von `0.2099`. Die anschließende Visualisierung im Scatterplot deckte jedoch ein systematisches Problem auf (siehe [mixup_initial_scatterplot.png](file:///Users/fietescheel/Documents/Master%20Thesis/research/img/analysis/mixup_initial_scatterplot.png)):

![Initial Scatterplot](img/analysis/mixup_initial_scatterplot.png)

- **Analyse:** Die Punkte bildeten eine flache horizontale Wolke um den Mittelwert (ca. 0.45). Echte Targets von `0.0` und `1.0` wurden fälschlicherweise auf Werte um `0.3` bzw. `0.6` geschätzt.
- **Grund:** Da der DataLoader bei jedem Aufruf zufällig mischte, veränderten sich die Validierungsdaten und deren Targets in jeder Epoche neu. Die Validierung war nicht deterministisch (es wurden Äpfel mit Birnen verglichen). Zudem verhinderte das Zerstören der sequentiellen Satzstruktur durch Shuffling, dass das BiLSTM sinnvolle Wort-Kontexte lernen konnte; das Modell entschied sich für die "sicherste" Vorhersage nahe dem globalen Mittelwert (Mean Prediction).

---

### 3.2. Lösung: Deterministische & Prä-generierte Datensätze

Um das Problem zu lösen, wurde die Dataset-Klasse `MixupPyTorchDataset` grundlegend umgeschrieben (Option A):

1. **Prä-Generierung:** Die Mischungen werden nun einmalig im Konstruktor (`__init__`) des Datasets erzeugt.
2. **Reproduzierbarkeit (Seeds):** Es wurde ein fester Random-Seed gesetzt (`42` für das Training, `99` für die Validierung), wodurch die Mischungsverhältnisse und Wortkombinationen über alle Epochen hinweg exakt gleich bleiben.
3. **Daten-Augmentierung:**
   - **Trainingsdaten:** 10 verschiedene feste Mischungen pro Artikelpaar $\rightarrow$ **9.280 feste Samples**.
   - **Validierungsdaten:** 2 feste Mischungen pro Artikelpaar $\rightarrow$ **208 feste Samples**.

---

### 3.3. Finale Ergebnisse & Interpretation

Das Modell wurde mit der korrigierten, deterministischen Pipeline trainiert. Zunächst über 20 Epochen und anschließend zur vollständigen Konvergenz über maximal 40 Epochen. Das Training wurde durch das Early-Stopping (Patience = 5) bei **Epoche 28** beendet.

Das beste Modell wurde in **Epoche 23** gespeichert und lieferte folgende finale Ergebnisse:

- **Abschließender Validation MSE:** **0.0335** (zuvor 0.0655, somit fast halbiert)
- **Abschließender Validation MAE:** **0.1195** (zuvor 0.2099, ebenfalls fast halbiert)

Der finale Scatterplot (siehe [mixup_final_scatterplot.png](file:///Users/fietescheel/Documents/Master%20Thesis/research/img/analysis/mixup_final_scatterplot.png)) zeigt die deutliche Verbesserung:

![Final Scatterplot](img/analysis/mixup_final_scatterplot.png)

#### Interpretation der Ergebnisse:

- **Erfolgreiches Lernen der Kontinuierlichen Kurve:** Die Punkte orientieren sich nun sehr stark an der diagonalen Linie (Perfekte Vorhersage $y = x$).
- **Präzision bei Extremwerten:** Echte Lambda-Werte von `0.0` (reine Alltagssprache) und `1.0` (reine Leichte Sprache) werden vom Modell mit hoher Sicherheit sehr nah an ihren echten Werten (`0.0` bis `0.2` bzw. `0.85` bis `1.0`) vorhergesagt.
- **MAE von ~0.12:** Im Durchschnitt weicht die Vorhersage des Modells nur um ca. 12 % vom echten Mischungsverhältnis ab. Dies beweist, dass das BiLSTM trotz der Zerstörung der Satzkohärenz (durch das Shuffling) in der Lage ist, die stilistische Komplexität der Sätze auf Wortebene zu erfassen und korrekt zu gewichten.

---

## 4. Nächste Schritte

1. **Evaluation auf LLM-Generierungen (Ansatz 2):** Test des Mix-Up-BiLSTMs auf den durch LLMs generierten Textstufen (`0.25`, `0.50`, `0.75`), um zu prüfen, ob die vom Klassifikator vorhergesagten Werte mit den LLM-Target-Stufen korrespondieren.
