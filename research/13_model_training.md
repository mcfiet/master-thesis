Hyperparameter Tuning zeitaufwändig und bringt nicht so viel

Regression
- Variante 1: Mix-Up (Satz von gleicher Länge jeweils von LS und AS)
- Variante 2: Mit LLM und LS und AS Texts als Input zwischen Steps

---

## Evaluation des BiLSTM-Modells auf einem proprietären Datensatz (Lebenshilfe)

Nach dem Training des BiLSTM-Klassifikators auf dem automatisiert erstellten Web-Korpus wurde die Generalisierungsfähigkeit des Modells ("Zero-Shot"-Performance) anhand eines komplett unabhängigen, unveröffentlichten Datensatzes der Lebenshilfe überprüft.

### 1. Datenaufbereitung (Lebenshilfe-Datensatz)
Die Ausgangsdaten bestanden aus 98 unstrukturierten Textdokumenten (Word-Dateien `.docx`, `.doc`, sowie `.odt` und `.rtf`), die manuelle Übersetzungen in Leichte Sprache (LS) und deren alltagssprachliche (AS) Originale enthielten. 
- **Sortierung & Zuordnung:** Durch ein automatisiertes Python-Skript (`create_lebenshilfe_dataset.py`) wurden die Dateinamen normalisiert (Entfernung von Tags wie "ILS", "Prüfer", "AD001"), um die zusammengehörigen AS- und LS-Paare zu matchen. Einige Paare mit stark abweichenden Dateinamen wurden manuell gemappt.
- **Textextraktion:** Der Rohtext wurde mittels `pandoc` sowie python-spezifischer Bibliotheken (`python-docx`, `odfpy`, `striprtf`) aus den Dokumenten extrahiert.
- **Ergebnis:** Es entstand ein bereinigter JSON-Datensatz (`results/lebenshilfe_dataset.json`) mit **49 verifizierten AS-LS-Artikelpaaren**.

### 2. Entwicklung einer universellen Evaluations-Pipeline
Um dieses (und zukünftige) Datensätze zu testen, wurde ein generisches Jupyter Notebook (`notebooks/evaluate_model_on_dataset.ipynb`) erstellt.
Das Setup umfasste:
- **Vokabular-Rekonstruktion:** Da das BiLSTM-Modell strikt an das Vokabular (ca. 25.000 Token) des Trainingsdatensatzes gebunden ist, rekonstruiert das Notebook zunächst dynamisch das Vokabular aus dem ursprünglichen, nach semantischer Ähnlichkeit gefilterten Trainings-CSV (`similarity 0.8 bis 0.98`).
- **Tokenisierung:** Nutzung des `de_core_news_sm` Spacy-Tokenizers (ohne POS/NER für maximale Performance). Sequenzen wurden auf eine Länge von 512 Token limitiert (Padding/Truncating).
- **Lesbarkeits-Metriken:** Zur Verifizierung der Modellvorhersagen durch linguistische Standardverfahren wurden Flesch Reading Ease (FRE) und die Wiener Sachtextformel integriert.

### 3. Testergebnisse (Zero-Shot Generalization)
Das beste existierende Modell (`lstm_article_sim_0.80_to_0.98.pt`) wurde ohne jegliches Fine-Tuning auf den 98 Lebenshilfe-Texten angewendet.

**Klassifikations-Metriken:**
- **Overall Accuracy:** 90.82% (89 von 98 Texten korrekt klassifiziert)
- **Balanced Accuracy:** 90.82%
- **Perfect Pair Match:** 81.6% (40 von 49 Textpaaren wurden einwandfrei unterschieden; d.h. LS als "Simple" UND das zugehörige AS-Original als "Normal" erkannt).

**Validierung durch klassische Lesbarkeitsmetriken (Readability):**
- **Avg LS Flesch:** 66.40 (Zielwert für Leichte Sprache > 80; dennoch deutlich im lesbaren Bereich)
- **Avg AS Flesch:** 43.29 (Formal/Schwer verständlich)
- **Flesch Gap:** 23.11 Punkte Differenz zwischen AS und LS.
- **Avg LS Wiener:** 7.55 (Entspricht ca. 7.-8. Klasse; Zielwert LS eigentlich < 6)
- **Avg AS Wiener:** 11.23 (Entspricht Gymnasialniveau)

### 4. Ausschluss von Layout-Biases (Absatz-Kontrollexperiment)
Ein potentieller Bias könnte darin bestehen, dass das Modell lernt, Leichte Sprache primär an der hohen Frequenz von Absätzen (kürzere Abschnitte, häufigere Zeilenumbrüche) zu erkennen. Dies wurde empirisch überprüft, indem eine absatzfreie Kontrollversion des Datensatzes (`lebenshilfe_dataset_no_paragraphs.json`) evaluiert wurde. Die Klassifikationsergebnisse und Konfidenzwerte blieben zu 100 % identisch (0 Abweichungen bei allen 98 Vergleichen). Dies lässt sich auch theoretisch begründen: Im Preprocessing des Tokenizers (`spacy.blank("de")`) werden sämtliche Whitespace-Tokens (`is_space`) herausgefiltert. Das BiLSTM erhält somit nur eine flache Wortsequenz. Ein Layout-Overfitting bezüglich der Absatzstruktur ist somit ausgeschlossen.

### 5. Interpretation und Diskussion der Ergebnisse
Das Ergebnis belegt eine herausragende Generalisierungsfähigkeit des Modells. 

**Vermeidung von Data Leakage:**
Die Trainingsdaten (Behörden- und Nachrichtenseiten) und die Testdaten (interne Dokumente, Hausordnungen, Satzungen der Lebenshilfe) überschneiden sich nicht. Das Modell hat diese Art von Textstruktur zuvor nie "gesehen". Die Accuracy von über 90% beweist empirisch, dass das BiLSTM echte linguistische Muster der Leichten Sprache erlernt hat und sich nicht an quellenspezifische Layouts ("Overfitting") klammert.

**Robustheit gegenüber neuem Vokabular:**
Viele juristische Fachbegriffe (z.B. in JVA-Hausordnungen) aus dem Lebenshilfe-Set tauchten im Web-Korpus nicht auf und wurden vom Modell als `<unk>` (Unknown) maskiert. Die hohe Genauigkeit zeigt, dass das Modell seine Entscheidung auf strukturelle Merkmale (Satzlängen, spezifische Konjunktionen, syntaktische Einfachheit) stützt, anstatt spezifische Vokabeln auswendig zu lernen.

**Mögliche Lücken / Biases (Diskussionspunkte für die Thesis):**
1. **Typografische Marker:** Leichte Sprache verwendet häufig Mediopunkte (`∙`) oder Bindestriche zur Silbentrennung (z.B. `Bewohner∙park∙zone`). Tokenizer behandeln diese oft als separate Token. Dies führt zu einer künstlich erhöhten Anzahl von "kurzen Wörtern" im Vektor. Das Modell könnte gelernt haben, dieses rein typografische Merkmal als starken Indikator für LS zu werten, was linguistisch nicht falsch, aber ein "Shortcut" (Bias) ist.
2. **Dokumentenlänge als Signal:** Obwohl die Sequenzen auf 512 Token begrenzt und mit Padding aufgefüllt wurden, sind LS-Texte durch starke inhaltliche Zusammenfassungen (Informationsverlust) generell kürzer. Dieser Bias wurde durch die Längen-Limitierung im Modell gemindert, sollte aber bei der Evaluation auf Artikel-Ebene immer mitbedacht werden.

**Fazit:**
Die Evaluation bestätigt die Verwendbarkeit eines maschinell und unsauber gecrawlten Web-Korpus zum Training robuster Klassifikatoren für Leichte Sprache. Das Modell ist in der Lage, professionelle, händisch erstellte Übersetzungen mit hoher Präzision zu identifizieren.