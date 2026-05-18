# 08 Dataset Analysis: Tiefergehende Analyse & Validierung

## 1. Status Quo und Problemstellung
Die initiale Analyse in `07_dataset_analysis.md` hat einen signifikanten Informationsverlust in Leichter Sprache (LS) aufgezeigt, insbesondere beim Erhalt von benannten Entitäten (NER Recall). Um diese Ergebnisse zu validieren und die Qualität der Metriken zu erhöhen, müssen technische Limitierungen der bisherigen Pipeline adressiert werden.

### 1.1 Limitation der SBERT-Sequenzlänge
### 1.1 Limitation der SBERT-Sequenzlänge & Coverage
Eine statistische Auswertung der Token-Anzahl im aktuellen Korpus hat ergeben, dass die anfängliche semantische Ähnlichkeitsanalyse unvollständig war. SBERT (MiniLM) hat ein Standardlimit von 128 Tokens, das wir im ersten Schritt auf 512 Tokens erhöht haben.

**Coverage-Analyse (Vergleich der Limits):**

| Abdeckung | AS (Limit: 128) | LS (Limit: 128) | | AS (Limit: 512) | LS (Limit: 512) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vollständig erfasst** | 9,4 % (144 Art.) | 5,1 % (78 Art.) | | **56,9 %** (869 Art.) | **52,9 %** (807 Art.) |
| **Abgeschnitten** | 90,6 % (1382 Art.)| 94,9 % (1448 Art.)| | **43,1 %** (657 Art.) | **47,1 %** (719 Art.) |

**Problem:** Beim ursprünglichen Limit von 128 Tokens wurden über 90% der Artikel abgeschnitten; gemessen wurden quasi nur die Einleitungen. Trotz der Erhöhung auf das Modell-Maximum von 512 Tokens werden weiterhin ca. **43-47% der Artikel abgeschnitten**. Dies unterstreicht die Notwendigkeit einer satzweisen Verarbeitung, um eine 100%ige Abdeckung zu erreichen.


## 2. Ergebnisse der optimierten Analyse (512 Tokens)

Durch die Erhöhung der `max_seq_length` von 128 auf **512 Tokens** konnte die semantische Ähnlichkeit deutlich präziser gemessen werden, da nun ein wesentlich größerer Teil der Artikel (bei vielen Quellen der gesamte Text) in die Berechnung einfließt.

### 2.1 Vergleich der semantischen Ähnlichkeit (SBERT)

Die folgende Tabelle zeigt die Auswirkung des erhöhten Token-Limits auf alle Quellen. Es wird deutlich, dass die semantische Ähnlichkeit fast überall steigt, wenn das Modell "mehr" vom Text lesen darf.

| Quelle | Similarity (128 Tokens) | Similarity (512 Tokens) | Differenz |
| :--- | :---: | :---: | :---: |
| **apotheken** | 0.636 | **0.894** | +0.258 |
| **behindertenbeauftragter** | 0.746 | **0.761** | +0.015 |
| **brandeins** | 0.637 | **0.698** | +0.061 |
| **hamburg** | 0.665 | **0.804** | +0.139 |
| **hannover** | 0.706 | **0.776** | +0.070 |
| **koeln** | 0.684 | **0.833** | +0.149 |
| **main_taunus** | 0.762 | **0.727** | -0.035 |
| **mdr** | 0.733 | **0.766** | +0.033 |
| **sozialpolitik** | 0.706 | **0.850** | +0.144 |
| **stuttgart** | 0.896 | **0.856** | -0.040 |
| **wiesbaden** | 0.750 | **0.642** | -0.108 |

**Interpretation:** 
Die massive Steigerung bei Quellen mit sehr langen Texten (wie *Apotheken Umschau* oder *Köln*) zeigt, dass die Einleitungen in Leichter Sprache oft stark vom Original abweichen (z. B. durch direkte Ansprache oder Leseanweisungen), während der Hauptteil der Erklärungen semantisch sehr nah am Original bleibt. Der anfänglich niedrige Ähnlichkeitswert war also teilweise ein Artefakt der Textkürzung (Truncation).

Interessanterweise sinkt bei drei Quellen (Main-Taunus, Stuttgart, Wiesbaden) die Ähnlichkeit bei längerer Betrachtung. Dies deutet darauf hin, dass hier die Einleitungen fast identisch sind, sich die Texte im weiteren Verlauf (z.B. durch starke Kürzungen in der LS) aber inhaltlich voneinander entfernen.

### 2.2 Visualisierung der Verteilung im Vergleich

Der Vergleich der Boxplots zeigt deutlich, wie die Erhöhung des Token-Limits das Bild der semantischen Ähnlichkeit verändert.

#### 1. Ursprüngliche Analyse (Limit: 128 Tokens)
*Fokus primär auf Artikeleinleitungen/Teaser.*

![Semantische Ähnlichkeit 128](../research/img/analysis/semantic_similarity_128.png)

#### 2. Optimierte Analyse (Limit: 512 Tokens)
*Einbeziehung von ca. 350-400 Wörtern pro Artikel.*

![Semantische Ähnlichkeit 512](../research/img/analysis/semantic_similarity_512.png)

**Beobachtung:** Während die Ähnlichkeit bei vielen Quellen (z.B. Apotheken, Köln) deutlich "nach oben wandert" und die Boxen kompakter werden, zeigt der 512-Token-Plot bei einigen Quellen (z.B. Wiesbaden) eine stärkere Streuung nach unten. Dies beweist, dass der Informationsverlust oft erst im weiteren Textverlauf massiv wird.

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
