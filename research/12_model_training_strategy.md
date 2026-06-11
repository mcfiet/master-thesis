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

## 3. Trainingskonfiguration

Für die ersten Trainingsläufe werden wir die erfolgreichsten Hyperparameter aus der Forschung übernehmen:

| Parameter | Wert |
| :--- | :--- |
| **Optimizer** | AdamW |
| **Lernrate** | $5 \times 10^{-5}$ |
| **Weight Decay** | 0,01 |
| **Batch Size** | 32 |
| **Max Epochen** | 30 (mit Early Stopping) |
| **Early Stopping Patience** | 3 - 5 |
| **LoRA Konfig (falls genutzt)** | r=50, alpha=32, dropout=0,0 |
| **LoRA Target Modules** | query, key, value, dense, proj |

## 4. Ausführungsplan

1.  **Datenvorbereitung:** Konvertierung des ausgerichteten Korpus (`results/corpus_final/`) in einen balancierten binären Klassifikationsdatensatz.
2.  **Baseline-Lauf:** Training der BiLSTM, um eine Untergrenze für die Performance festzulegen.
3.  **Vollständiges Fine-Tuning:** Durchführung eines vollständigen Fine-Tunings von SBERT auf dem zusammengeführten Korpus.
4.  **Evaluierung:** Verwendung von Balanced Accuracy, F1-Score und qualitativer Analyse von Fehlklassifikationen zur Bewertung der Modellleistung.
5.  **Validierung auf Realdaten:** Da das Training möglicherweise noch auf synthetischen oder semi-synthetischen Daten basiert, ist die Validierung gegen einen hand-aligned Testset entscheidend.

## 5. Erste Ergebnisse (Master-Thesis Korpus)

Im ersten Durchlauf wurde die BiLSTM-Baseline auf dem aktuell vorliegenden Korpus (`results/corpus_final/`) trainiert, um eine empirische Basislinie für diese spezifischen Daten zu erhalten.

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

