# Modell-Trainingsstrategie

Basierend auf den Erkenntnissen aus vorherigen Experimenten im `genai-project` skizziert dieses Dokument die Strategie für das Training der ersten Klassifikationsmodelle für den Korpus "Leichte Sprache".

## 1. Erkenntnisse aus dem `genai-project`

Im `genai-project` wurden verschiedene Trainingsparadigmen für die binäre Klassifikation (Einfaches vs. Normales Deutsch) unter Verwendung von Sentence-BERT (SBERT) und LSTM-Baselines verglichen.

### 1.1 Performance-Vergleich (Balanced Accuracy)

| Modellkonfiguration | Balanced Accuracy (BAcc) | Epochen bis Konvergenz |
| :--- | :---: | :---: |
| **SBERT Vollständiges Fine-Tuning** | **0,913 - 0,947** | 4 - 5 |
| **SBERT + LoRA (r=50, alpha=32)** | **0,906** | 10 |
| **BiLSTM Baseline** | 0,863 | 7 |
| SBERT Letzte Schicht frei | 0,834 | 11 |
| SBERT Eingefroren (MLP-Kopf) | 0,763 | 23 |
| SBERT Eingefroren (Linearer Kopf) | 0,719 | 100+ |

### 1.2 Trainingsdauer und Effizienz

| Modellkonfiguration | Minuten bis beste Val-Acc | Validierungs-Accuracy |
| :--- | :---: | :---: |
| **LSTM-Baseline** | **0,27** | 0,863 |
| SBERT Letzte Schicht frei | 9,35 | 0,827 |
| **SBERT Vollständiges Fine-Tuning** | **11,01** | 0,903 |
| SBERT Eingefroren (MLP-Kopf) | 15,71 | 0,760 |
| **SBERT + LoRA** | 20,24 | 0,902 |
| SBERT Eingefroren (Linearer Kopf) | 82,70 | 0,707 |

### 1.3 Zentrale Erkenntnisse

1.  **Das Einfrieren des Encoders ist unzureichend:** Die Verwendung von SBERT lediglich als Feature-Extraktor (eingefroren) führt zu unterdurchschnittlichen Ergebnissen (max. 0,76 BAcc). Das Modell benötigt eine End-to-End-Anpassung, um die Nuancen der Leichten Sprache zu erfassen.
2.  **Vollständiges Fine-Tuning ist hochwirksam:** Trotz der hohen Parameterzahl konvergiert es schnell (ca. 5–11 Minuten auf dem synthetischen Datensatz) und liefert die beste Genauigkeit.
3.  **LoRA als starke Alternative:** LoRA (Low-Rank Adaptation) bietet eine Performance, die sehr nah am vollständigen Fine-Tuning liegt, während die Anzahl der trainierbaren Parameter signifikant reduziert wird. Dies ist die bevorzugte Wahl, falls der GPU-Speicher zum Flaschenhals wird.
4.  **Sequenzieller Kontext ist wichtig:** Die BiLSTM-Baseline übertrifft die eingefrorenen SBERT-Modelle. Dies deutet darauf hin, dass die vortrainierten Embeddings allein (ohne Fine-Tuning) die für Leichte Sprache typischen strukturellen Vereinfachungen nicht vollständig abbilden.
5.  **Datenskalierung:** Die Genauigkeit verbessert sich kontinuierlich, wenn der synthetische Datensatz von ~30k auf ~190k Beispiele anwächst.

## 2. Erste Modellauswahl

### 2.1 Primäres Modell: Sentence-BERT (Multilingual)
Wir werden weiterhin **`paraphrase-multilingual-MiniLM-L12-v2`** als primäres Backbone verwenden. Es ist effizient, unterstützt Deutsch und hat exzellente Fine-Tuning-Eigenschaften gezeigt.

### 2.2 Sekundäre Baseline: BiLSTM
Eine einfache bidirektionale LSTM wird als leichtgewichtige Baseline dienen, um zu überprüfen, ob die Transformer-Architektur für die spezifische Komplexität unseres Korpus notwendig ist.

## 3. Trainingskonfigurationen

Um optimale Ergebnisse zu erzielen, unterscheiden wir in der Konfiguration zwischen der BiLSTM-Baseline und dem späteren SBERT Fine-Tuning.

### 3.1 Konfiguration: BiLSTM-Baseline (verwendet für 99% BAcc)

| Parameter | Wert |
| :--- | :--- |
| **Optimizer** | AdamW |
| **Lernrate** | $1 \times 10^{-3}$ |
| **Weight Decay** | 0,01 |
| **Batch Size** | 32 |
| **Max Epochs** | 30 |
| **Early Stopping Patience** | 7 |
| **Dropout** | 0,4 |
| **Max Seq Len** | 512 (Artikel) / 100 (Sätze) |

### 3.2 Ziel-Konfiguration: SBERT Fine-Tuning

| Parameter | Wert |
| :--- | :--- |
| **Optimizer** | AdamW |
| **Lernrate** | $5 \times 10^{-5}$ |
| **Weight Decay** | 0,01 |
| **Batch Size** | 32 |
| **Max Epochs** | 10 - 20 |
| **Early Stopping Patience** | 3 |
| **LoRA Konfig (optional)** | r=50, alpha=32 |
| **Max Seq Len** | 512 |

## 4. Ausführungsplan

1.  **Datenvorbereitung:** Konvertierung des ausgerichteten Korpus (`data/corpus/final/`) in einen balancierten binären Klassifikationsdatensatz.
2.  **Baseline-Lauf:** Training der BiLSTM, um eine Untergrenze für die Performance festzulegen.
3.  **Vollständiges Fine-Tuning:** Durchführung eines vollständigen Fine-Tunings von SBERT auf dem zusammengeführten Korpus.
4.  **Evaluierung:** Verwendung von Balanced Accuracy, F1-Score und qualitativer Analyse von Fehlklassifikationen zur Bewertung der Modellleistung.
5.  **Validierung auf Realdaten:** Da das Training möglicherweise noch auf synthetischen oder semi-synthetischen Daten basiert, ist die Validierung gegen einen hand-aligned Testset entscheidend.

## 5. Erste Ergebnisse (Master-Thesis Korpus)

Im ersten Durchlauf wurde die BiLSTM-Baseline auf dem aktuell vorliegenden Korpus (`data/corpus/final/`) trainiert, um eine empirische Basislinie für diese spezifischen Daten zu erhalten.

### 5.1 BiLSTM Baseline Ergebnisse

| Metrik | Wert |
| :--- | :--- |
| **Datensatz-Größe** | ~118.000 Sätze (balanciert) |
| **Vokabular-Größe** | 20.002 Tokens |
| **Balanced Accuracy** | **94,12 %** |
| **F1-Score (Normal / Einfach)** | 0,94 / 0,94 |
| **Trainingszeit** | ~1 Minute / Epoche (20 Epochen) |

### 5.2 Einordnung
Die Ergebnisse übertreffen die Baseline aus dem `genai-project` (86,3 %) deutlich. Dies liegt vermutlich an der höheren Qualität und dem größeren Umfang des bereinigten Korpus. Eine Balanced Accuracy von über 94 % zeigt, dass die sprachlichen Unterschiede (Satzlänge, Wortwahl, Struktur) zwischen Normaler und Leichter Sprache im vorliegenden Datensatz sehr markant sind und sich bereits mit einfachen sequenziellen Modellen zuverlässig klassifizieren lassen.

## 6. Satz-Level Klassifikation und Similarity-Analyse

Zunächst wurde untersucht, wie gut die Klassifikation auf Basis einzelner Sätze funktioniert. Dies ermöglichte eine feingliedrige Analyse des Einflusses der Alignment-Qualität (Similarity) auf die Modellgüte.

### 6.1 Ergebnisse der Satz-Filterung

| Similarity Bereich | Artikelpaare | Sätze (balanciert) | Balanced Accuracy |
| :--- | :---: | :---: | :---: |
| 0,60 - 0,98 | 1474 | ~29.700 | 92,48 % |
| 0,70 - 0,98 | 1362 | ~27.700 | 92,43 % |
| **0,80 - 0,98** | **1032** | **~21.400** | **92,99 %** |
| 0,90 - 0,98 | 213 | ~4.800 | 90,55 % |

*Hinweis: Der Bereich wurde bei 0,98 gedeckelt, um identische Artikelpaare (Dubletten) auszuschließen.*

### 6.2 Analyse der Satz-Ebene
Die Satz-Ebene lieferte die erste wichtige Erkenntnis: Der Bereich **0,80 - 0,98** ist der qualitative "Sweet Spot". Trotz der hohen Genauigkeit von ~93 % zeigten sich jedoch Grenzen bei sehr kurzen oder isolierten Sätzen, denen der stilistische Kontext fehlte.

## 7. Aufbauend: Artikel-Level Klassifikation

Aufbauend auf der Erkenntnis, dass der Bereich 0,80 - 0,98 die beste Datenbasis darstellt, wurde untersucht, ob die Einbeziehung des gesamten Artikel-Kontexts (bis zu 512 Tokens) die Unterscheidungskraft weiter schärft.

### 7.1 Ergebnisse der Artikel-Klassifikation

| Similarity Bereich | Artikel (gesamt) | Balanced Accuracy |
| :--- | :---: | :---: |
| 0,60 - 0,98 | 2944 | 95,93 % |
| 0,70 - 0,98 | 2720 | 97,30 % |
| **0,80 - 0,98** | **2060** | **99,03 %** |
| 0,90 - 0,98 | 426 | 98,44 % |

### 7.2 Interpretation der Steigerung
Der Sprung von **93 % auf 99 %** Balanced Accuracy im optimalen Bereich bestätigt, dass die Klassifikation von "Leichter Sprache" eine holistische Aufgabe ist. Während einzelne Sätze bereits starke Signale liefern, summieren sich die stilistischen Merkmale auf Artikel-Ebene zu einer nahezu fehlerfreien Vorhersage auf.

## 8. Fazit für die weitere Modellentwicklung

Die Experimente zeigen konsistent, dass der Similarity-Bereich **0,80 - 0,98** die qualitativ hochwertigste Basis für das Training darstellt. Während die Satz-Ebene für granulare Analysen nützlich ist, bietet die Artikel-Ebene eine nahezu fehlerfreie Klassifikation.

**Nächster Schritt:** 

- Test Set bauen / analysieren von Lebenshilfe Kiel (In früheren Projekten ist da die Acc abgefallen)
- BiLSTM mit höherer Input Größe trainieren lassen (aktuell 512 Tokens, weil gebräuchlich)
- Transformer basierte Encoder verwenden, mit denen sich dann größere Input Lengths abbildern lassen (512 to 8,192 tokens)

**Fragen**

- Wörter die weniger als 3 mal vorkommen raus um rauschen zu verhindern. macht das sinn?

Wenn seltene Wörter zu <unk> umgewandelt werden, kann das Modell lernen: "Ein Text mit vielen <unk>-Tokens ist wahrscheinlich Alltagssprache". Das <unk>-Token selbst wird also zu einem nützlichen Feature für den Klassifikator, um die Komplexität eines Artikels zu bewerten.

- 

Moving-Average Type-Token Ratio