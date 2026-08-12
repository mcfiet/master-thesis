# Regressions-Implementierung & Evaluierungs-Setup

In diesem Dokument wird die konkrete technische Umsetzung der beiden Regressionsansätze (Satzebenen-Mix-Up und LLM-basierte Generierung synthetischer Zwischenstufen) dokumentiert, die in Woche 15 durchgeführt wurden.

---

## 1. Ansatz 1: Sentence-Level Mix-Up (Erste Variante)

Ansatz 1 zielt darauf ab, ohne zusätzliche Generierungskosten kontinuierliche Komplexitätsübergänge zu erzeugen, indem Sätze aus Leichter Sprache (LS) und Alltagssprache (AS) gemischt werden.

### Technische Umsetzung

Wir haben die Klasse `NormalMixupDataset` in `scripts/run_mixup_test.py` implementiert. Der Ablauf gestaltet sich wie folgt:

1. **Satzsegmentierung:** Die verifizierten Artikelpaare werden mittels `spaCy` (`sentencizer`) in einzelne Sätze zerlegt.
2. **Zufälliges Slicing:** Für LS und AS wird jeweils unabhängig ein zufälliger, zusammenhängender Bereich (Start- und End-Index) ausgewählt.
3. **Mischung & Shuffle:** Die beiden Slices werden zusammengeführt und zufällig gemischt, um die Absatzstruktur aufzubrechen.
4. **Target-Berechnung:** Das Regressionstarget $\lambda$ berechnet sich dynamisch als das Verhältnis der Zeichenlänge des LS-Anteils zur Gesamtzeichenlänge des erzeugten Absatzes.

#### Pseudo-Code der Implementierung:

```python
# 1. Satzsegmentierung
sents_ls = [s.text.strip() for s in nlp_sentencizer(ls_text) if s.text.strip()]
sents_as = [s.text.strip() for s in nlp_sentencizer(as_text) if s.text.strip()]

# 2. Unabhängige Auswahl zufälliger contiguous Slices
start_ls, end_ls = sorted([random.randint(0, len(sents_ls)), random.randint(0, len(sents_ls))])
sample_ls = sents_ls[start_ls:end_ls]

start_as, end_as = sorted([random.randint(0, len(sents_as)), random.randint(0, len(sents_as))])
sample_as = sents_as[start_as:end_as]

# 3. Zusammenführen und Shuffeln
mixed_paragraph = sample_ls + sample_as
random.shuffle(mixed_paragraph)

# 4. Target berechnen (Verhältnis der Zeichenlängen)
char_len_ls = len("".join(sample_ls))
char_len_as = len("".join(sample_as))
total_char_len = char_len_ls + char_len_as

regression_target = char_len_ls / total_char_len if total_char_len > 0 else 0.5
```

### Analyse der Target-Verteilung

Um die Verteilung der Regressionstargets zu überprüfen, wurde eine Simulation über 10 Epochen auf Basis des bereinigten Datensatzes durchgeführt (1.032 Paare). Die Ergebnisse wurden in [mixup_first_variant_distribution.png](file:///Users/fietescheel/Documents/Master%20Thesis/research/img/analysis/mixup_first_variant_distribution.png) visualisiert.

- **Ergebnis:** Die Target-Verteilung zeigt eine Häufung um den Mittelwert `0.5` sowie an den Rändern (`0.0` und `1.0`).
- **Bewertung:** Die Verteilung ist für das Training verwendbar, da davon auszugehen ist, dass das neuronale Regressionsmodell robust genug reagiert. Die Häufungen sind nicht zu extrem, sollten jedoch während des Trainings beobachtet werden.

### Konzeptuelle Alternative (Zweite Variante als Backup)

Sollte sich im Training herausstellen, dass die Häufung um `0.5` und an den Extremen zu stark ausgeprägt ist und die Regressionsgenauigkeit beeinträchtigt, steht die **Zweite Variante** als Backup-Idee zur Verfügung:

- Vorab wird ein gleichverteiltes Target $\lambda \sim U(0.0, 1.0)$ gezogen.
- Basierend auf einer festen Absatzgröße $N$ (z. B. 20 Sätze) wird die exakte Satzanzahl bestimmt: $num\_ls = \text{round}(\lambda \cdot N)$ und $num\_as = N - num\_ls$.
- Es werden zusammenhängende Slices dieser exakten Längen gezogen. Dies erzwingt eine perfekte Gleichverteilung der Regressionstargets über das Intervall $[0.0, 1.0]$.

---

## 2. Ansatz 2: LLM-basierte Zwischenstufen (Ausführung & Bereinigung)

Ansatz 2 nutzt LLMs, um semantisch konsistente und stilistisch saubere Texte auf den Ziel-Komplexitätsstufen `0.25`, `0.50` und `0.75` zu generieren.

### Ausführung der Pipeline

Das Skript `generate_synthetic_regression_steps.py` wurde auf zwei Plattformen getestet und ausgeführt:

1. **Lokal (Ollama):** Erfolgreicher Testlauf mit `LLaMA 3` (8B) via `localhost:11434` zur Verifizierung der JSON-Ausgabestruktur.
2. **Server (GPU):** Ausführung mit dem 120B-Modell `FlensGen-GPT-OSS-120B` via HTTP-Endpunkt (`193.175.188.202:8000`), um eine höhere sprachliche Qualität und Konsistenz zu erreichen. Hierzu musste die Cisco AnyConnect VPN-Verbindung aktiv sein.

### Technische Anpassungen & Lessons Learned

1. **Regex-basiertes Post-Processing:**
   - _Problem:_ Das Modell neigte trotz systematischer Instruction im System-Prompt zu einleitenden Sätzen (z. B. _"Hier ist der Text auf Stufe 0.25..."_).
   - _Lösung:_ Ergänzung der Pipeline um ein Post-Processing-Skript, das solche Standardphrasen mittels regulärer Ausdrücke automatisiert erkennt und abschneidet.
2. **Erhalt der Layoutstrukturen:**
   - _Problem:_ Bei Stufe `0.25` (nahe an Leichter Sprache) wurden Zeilenumbrüche und Aufzählungspunkte in Fließtext konvertiert.
   - _Lösung:_ Anpassung der Prompt-Gewichtung, um die Strukturierung (kurze Zeilen, Bullets) auf niedrigen Stufen explizit zu fordern.
3. **Korrektur von Daten-Mismatches (Alignment-Fehler):**
   - _Problem:_ Bei der manuellen Durchsicht fiel auf, dass der erste Datensatz-Eintrag in `lebenshilfe_dataset.json` fälschlicherweise das LS-Dokument zum Geologiemuseum mit einer AS-Pressemitteilung zum CAU-Aktionstag verknüpfte.
   - _Lösung:_ Korrektur der Zuordnungen im Skript `create_lebenshilfe_dataset.py`, um Fehlgenerierungen des LLMs durch unpassende Kontexte zu verhindern.

---

## 3. Nächste Schritte

1. **Dataloader-Training:** Implementierung des Modelltrainings (MLP auf SBERT-Embeddings bzw. BiLSTM) unter Verwendung des Mix-Up-Dataloaders.
2. **Remote-Aufruf beheben:** Analyse des Verbindungsproblems zum remote GPU-Server (Modell reagiert nicht auf API-Anfragen, obwohl der Server pingbar ist).
