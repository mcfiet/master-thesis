# 11 Dataset Analysis - Next Steps

Hier sind die nächsten Schritte für die Analyse:

- [ ] **Stiftung Lesen:** Korpus mit deren Kriterien für Leichte Sprache abgleichen. Oder weitere Quellen?
- [x] **Lesbarkeitsindex:** Klassische Metriken (wie Flesch) für die Texte berechnen. Als Unterstützung und Vergleich
- [x] **Type-Token-Ratio:** Vielfalt des Wortschatzes prüfen (Wortwiederholungen). (MATTR Reduktion um ca. 13.6% in LS)
- [ ] **Satzlänge:** Durchschnittliche Länge der Sätze in AS vs. LS auswerten. Auch ob es über Quellen hinweg sich unterscheided.
- [ ] **Ratio-Unterschiede:** Warum ist das Verhältnis von AS zu LS je nach Quelle so anders? (Vermutung: Varianz hilft dem Modell beim Lernen).
- [ ] **Cosinus-Ähnlichkeit:** Verschiedene Varianten/Modelle zur Inhaltsähnlichkeit testen.
- [ ] **Alpha dekar:** (Reliabilität/Güte prüfen).
- [ ] **Regressions-Metrik:** Eine Metrik entwickeln, die feiner als "richtig/falsch" ist (als Reward für das spätere Training).

## Ergebnisse der Analysen

### Type-Token-Ratio (TTR)
Die Analyse der lexikalischen Vielfalt mittels MATTR (Moving Average Type-Token-Ratio, Fenstergröße 50, lemmatisiert) ergab folgende Erkenntnisse:
- **Gesamtreduktion:** Die Wortvielfalt sinkt in LS um durchschnittlich **13,6 %** im Vergleich zu AS.
- **Durchschnittswerte:** AS (0,778) vs. LS (0,672).
- **Spitzenreiter der Vereinfachung:** *Hannover* (0,656) und *Hamburg* (0,658) weisen die geringste lexikalische Varianz in LS auf.
- **Journalistische Einflüsse:** Die *taz* (0,742) zeigt die höchste Varianz in LS, was auf ein höheres Sprachniveau innerhalb der Leichten Sprache hindeutet.
- **Methodik:** Durch die Nutzung von MATTR und Lemmatisierung konnte der verzerrende Effekt der unterschiedlichen Textlängen eliminiert werden.

**Visualisierungen:**
![MATTR Vergleich](img/analysis/ttr_mattr_comparison.png)
*Abbildung: Vergleich der lexikalischen Vielfalt (MATTR) nach Quelle.*

![TTR vs Length Scatter](img/analysis/ttr_vs_length_scatter.png)
*Abbildung: TTR in Abhängigkeit von der Textlänge (log-Skala) mit Regressionslinien.*
