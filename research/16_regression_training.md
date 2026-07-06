# Regressions-Training & Mix-Up-Evaluierung (Woche 16)

In dieser Woche (Woche 16) wird mit dem Training des Regressionsmodells auf Basis des Mix-Up-Ansatzes begonnen. Zu Beginn wurden die zusammengestellten Absätze aus dem Mix-Up-Dataloader ausgegeben und analysiert, um die Qualität und Struktur der erzeugten Trainingsdaten manuell zu überprüfen.

---

## 1. Analyse der gemischten Absätze (Mix-Up-Dataloader)

Der Mix-Up-Dataloader schneidet unabhängig voneinander contiguous (zusammenhängende) Slices aus den Leichte-Sprache- (LS) und Alltagssprache- (AS) Versionen eines Artikels aus, führt diese zusammen, shuffelt sie und berechnet das Regressionstarget $\lambda$ als Verhältnis der Zeichenlänge des LS-Anteils zur Gesamtlänge.

### Beispiel eines zusammengestellten Mix-Up-Absatzes

Aus dem Notebook [3_mixup_dataloader_test.ipynb](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/3_mixup_dataloader_test.ipynb) wurde das folgende Beispiel generiert und analysiert:

#### LS-Sätze (Extrakt, $n = 2$):
- *Die Beauftragten der Bundes-Regierung für die Belange von Menschen mit Behinderungen haben viele Aufgaben.*
- *Was macht der Behindertenbeauftragte der Bundesregierung?*

#### AS-Sätze (Extrakt, $n = 5$):
- *Inhaltsverzeichnis Video: Was macht der Behindertenbeauftragte der Bundesregierung?*
- *Gesetzlicher Auftrag Politische und soziale Rahmenbedingungen mitgestalten Informieren – beraten – Öffentlichkeitsarbeit leisten – Inklusionsgedanken verbreiten Grenzen der Beratung Video: Was macht der Behindertenbeauftragte der Bundesregierung?*
- *Video: Was macht der Behindertenbeauftragte der Bundesregierung?*
- *zum Download: Video: Was macht der Behindertenbeauftragte der Bundesregierung?*
- *(307 MB, 02:39) Gesetzlicher Auftrag Der/Die Behindertenbeauftragte wird vom Bundeskabinett jeweils für die Dauer einer Legislaturperiode bestellt.*

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

## 3. Nächste Schritte

1. **Trainingsskript aufsetzen:** Implementierung des Modelltrainings (MLP auf SBERT-Embeddings bzw. BiLSTM) zur Vorhersage des kontinuierlichen Werts $\lambda$.
2. **Evaluationsmetriken integrieren:** Automatisierter Vergleich der Regressionsvorhersagen mit klassischen Lesbarkeitsindizes.
