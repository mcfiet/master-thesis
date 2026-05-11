# 08 Dataset Analysis: Tiefergehende Analyse & Validierung

## 1. Status Quo und Problemstellung
Die initiale Analyse in `07_dataset_analysis.md` hat einen signifikanten Informationsverlust in Leichter Sprache (LS) aufgezeigt, insbesondere beim Erhalt von benannten Entitäten (NER Recall). Um diese Ergebnisse zu validieren und die Qualität der Metriken zu erhöhen, müssen technische Limitierungen der bisherigen Pipeline adressiert werden.

### 1.1 Limitation der SBERT-Sequenzlänge
Eine statistische Auswertung der Token-Anzahl im aktuellen Korpus hat ergeben, dass die bisherige semantische Ähnlichkeitsanalyse unvollständig war:

**Token-Statistik (N=1.526 Artikelpaare):**
| Metrik | Alltagssprache (AS) | Leichte Sprache (LS) |
| :--- | :---: | :---: |
| **Median (50%)** | **431 Tokens** | **479 Tokens** |
| **Maximum** | 8.262 Tokens | 3.386 Tokens |
| **75% Quartil** | 783 Tokens | 855 Tokens |

**Problem:** Das verwendete SBERT-Modell (`paraphrase-multilingual-MiniLM-L12-v2`) ist standardmäßig auf eine **Maximum Sequence Length von 128 Tokens** eingestellt. Da mehr als 75% der Texte länger als 128 Tokens sind, wurde bisher nur die Ähnlichkeit der Artikeleinleitungen gemessen. Der Rest der Artikel wurde abgeschnitten (Truncation).

## 2. Ergebnisse der optimierten Analyse (512 Tokens)

Durch die Erhöhung der `max_seq_length` von 128 auf **512 Tokens** konnte die semantische Ähnlichkeit deutlich präziser gemessen werden, da nun ein wesentlich größerer Teil der Artikel (bei vielen Quellen der gesamte Text) in die Berechnung einfließt.

### 2.1 Vergleich der semantischen Ähnlichkeit (SBERT)

| Quelle | Similarity (128 Tokens) | Similarity (512 Tokens) | Differenz |
| :--- | :---: | :---: | :---: |
| **apotheken** | 0.64 | **0.89** | +0.25 |
| **koeln** | 0.68 | **0.83** | +0.15 |
| **hannover** | 0.71 | **0.78** | +0.07 |
| **sozialpolitik** | 0.71 | **0.85** | +0.14 |

**Interpretation:** Die massive Steigerung bei den *Apotheken*-Artikeln zeigt, dass die Einleitungen in Leichter Sprache oft stark vom Original abweichen (z. B. durch direkte Ansprache oder Leseanweisungen), während der Hauptteil der medizinischen Erklärungen semantisch sehr nah am Original bleibt. Der ursprüngliche niedrige Wert war also teilweise ein Artefakt der Textkürzung (Truncation).

### 2.2 Visualisierung der Verteilung
Die folgende Grafik zeigt die Verteilung der semantischen Ähnlichkeit über alle Quellen hinweg nach der Optimierung auf 512 Tokens:

![Semantische Ähnlichkeit nach Quelle](../research/img/analysis/semantic_similarity_by_source.png)
*Abbildung 1: Boxplot der semantischen Ähnlichkeit (SBERT, 512 Tokens).*

### 2.3 Fazit zum Faktenerhalt vs. Semantik
Obwohl die **semantische Ähnlichkeit** durch die Einbeziehung von mehr Kontext deutlich gestiegen ist, bleibt der **NER Recall (Faktenerhalt)** weiterhin kritisch niedrig (oft < 20%). 

**Zentrale Erkenntnis:** Leichte Sprache transportiert die *Botschaft* und den *Sinn* des Textes (hohe Similarity) über weite Strecken erfolgreich, verzichtet dabei aber konsequent auf *konkrete Details und Fakten* (niedriger NER Recall).

## 3. Geplante Erweiterungen & Validierungsschritte (Update)

### 2.1 Optimierung der Semantischen Ähnlichkeit
- **Anpassung:** Umgehung des Token-Limits durch Zerlegung der Artikel in Sätze (Sentence Splitting).
- **Methode:** Berechnung der Embeddings für jeden Satz einzeln und anschließende Mittelwertbildung (Mean Pooling) über den gesamten Artikel.
- **Ziel:** Erfassung der semantischen Ähnlichkeit über die volle Textlänge.

### 2.2 Bidirektionale NER-Analyse
- **Hypothese:** Der Informationsverlust ist eine Einbahnstraße (AS -> LS).
- **Test:** Durchführung der NER-Analyse in beide Richtungen:
    1. **AS -> LS (Recall):** Wie viele Fakten bleiben erhalten? (Bisheriger Stand: ~20%).
    2. **LS -> AS:** Sind alle Informationen aus dem LS-Text auch im AS-Text enthalten? 
- **Ziel:** Bestätigung, dass LS-Texte zwar Informationen weglassen, aber keine *neuen* (fiktiven) Fakten hinzufügen, die nicht im Original stehen.

### 2.3 Manuelle Auditierung der Extremwerte
- **Fokus:** Analyse von Artikelpaaren mit extrem niedriger oder extrem hoher semantischer Ähnlichkeit.
- **Prüfung:** Handelt es sich um echte inhaltliche Abweichungen oder um Fehler im Scraping/Alignment?
- **Ziel:** Bereinigung des Korpus von "Noise" vor der Modelltrainingsphase.

### 2.4 Korrelationsanalyse: Token-Ratio vs. Similarity
- **Fragestellung:** Führt eine höhere Token-Ratio (mehr Erklärungen in LS) zwangsläufig zu einer höheren semantischen Ähnlichkeit zum Original?
- **Ziel:** Verständnis der Balance zwischen Vereinfachung (Kürzung) und Elaboration (Erklärung).

## 3. Zeitplan & Organisation
- [ ] Implementierung der verbesserten Metriken (Satzweises SBERT & Bidirektionales NER).
- [ ] Durchführung der re-evaluierten Analyse über das gesamte Korpus.
- [ ] Manueller Check der Top/Bottom 5% Ähnlichkeits-Ausreißer.
- [ ] Vorbereitung der Ergebnisse für das Treffen mit den Betreuern.
