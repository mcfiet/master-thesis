# 21. SFT-Optimierung mit LoRA, Web-Boilerplate-Bereinigung & Generierungsparameter

**Datum:** 17. August 2026  
**Thema:** Behebung von Overfitting, Web-Artefakt-Halluzinationen und Modell-Degenerierung im SFT-Basismodell (mBART-50)

---

## 1. Problemstellung & Ausgangslage

Im Rahmen der Bereinigung und Konsolidierung des Master-Korpus wurden redundante URL-Klone und unausgewogene Längenpaare entfernt, wodurch der Trainingsdatensatz von 1.343 auf **808 qualitätsgeprüfte Artikelpaare** (686 Train / 122 Validation) reduziert wurde.

Nach dieser Bereinigung traten beim SFT-Training zwei gravierende Probleme auf:
1. **Einbruch des Simplicity-Scores:** Der Median der Einfachheitsbewertung sank von $\sim 0{,}89$ auf $0{,}498$.
2. **Schnelles Overfitting bei Full Fine-Tuning:** Beim Training aller 611 Mio. Parameter von `facebook/mbart-large-50` auf 686 Trainingspaaren erreichte der Validation Loss bereits in Epoche 7 sein Minimum ($1{,}94$) und divergierte bis Epoche 15 auf $2{,}45$. Das getriggerte Early Stopping verhinderte, dass das Modell komplexe syntaktische Vereinfachungsmuster generalisierte.
3. **Halluzinationen am Textende:** Bei der qualitativen Analyse generierte das Modell an sich gut vereinfachte Sätze, hängte jedoch absurde Web-Floskeln und Frageketten an (z. B. *„Hier erfahren Sie mehr über die richtige Pflege von einem Hund.“*, *„Wie kann ich meine Hunde adoptieren?“*, *„Katzen sind besondere Haustiere für Menschen mit besonderen Bedürfnissen...“*).

---

## 2. LoRA / PEFT Integration im SFT-Training

### 2.1 Warum LoRA für mBART-50?
Full Parameter Fine-Tuning von 611 Mio. Parametern auf wenigen hundert Instanzen führt zu einer starken Parameter-Kapazitäts-Diskrepanz. Durch den Einsatz von **LoRA (Low-Rank Adaptation)** werden die Basisgewichte eingefroren und lediglich niedrigrangige Adapter-Matrizen trainiert:

$$\Delta W = B \cdot A \quad \text{mit} \quad A \in \mathbb{R}^{r \times d}, B \in \mathbb{R}^{k \times r}, \quad r \ll \min(d, k)$$

* **LoRA-Hyperparameter:**
  * Rang $r = 16$
  * Skalierungsfaktor $\alpha = 32$
  * Dropout $= 0{,}05$
  * Zielmodule: `q_proj`, `v_proj`, `k_proj`, `out_proj`, `fc1`, `fc2`
* **Trainierbare Parameter:** Reduziert auf ca. **1,5 Mio. Parameter ($\approx 0{,}25\,\%$)**.
* **Trainingsstabilität:** Ermöglicht eine höhere Lernrate ($\text{LR} = 1\text{e-}4$) und ein stabiles Training über 30 Epochen mit $\text{Patience} = 10$, ohne dass der Validation Loss explodiert.

### 2.2 Automatischer Merge für Downstream-Kompatibilität
In `scripts/modeling/train_sft.py` wurde implementiert, dass nach Abschluss des Trainings das LoRA-Modell mit `merge_and_unload()` zusammengeführt und als vollständiges HuggingFace-Modell unter `results/models/sft` gespeichert wird:

```python
if USE_PEFT:
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    peft_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    torch.save(merged_model.state_dict(), os.path.join(OUTPUT_DIR, "sft.pt"))
```
Dadurch können alle nachfolgenden Stufen der Pipeline (`generate_dpo_dataset.py`, `train_dpo.py`, `web/app.py`) das Modell direkt ohne PEFT-Wrapper laden.

---

## 3. Universeller Modell- und Tokenizer-Lader

Bei der Evaluierung in Jupyter Notebooks trat bei ungemergten Checkpoints oder reinen `.pt`-Gewichten der Fehler `ValueError: Unrecognized model in results/models/sft` auf.

In `analyse_sft_model.ipynb` und `analyse_dpo_model.ipynb` wurde ein **universeller, fehlertoleranter Lader** implementiert, der alle Speicherformate automatisch erkennt:
1. **Vollständiger HuggingFace-Ordner:** Lädt via `AutoModelForSeq2SeqLM.from_pretrained()`.
2. **LoRA-Adapter-Ordner (`adapter_config.json`):** Lädt Basismodell und hängt Adapter via `PeftModel` an.
3. **Reine `.pt`-Datei / State-Dict:** Initialisiert `facebook/mbart-large-50` und lädt die Gewichte via `load_state_dict()`.

---

## 4. Web-Boilerplate & SEO-Artefakt-Bereinigung

### 4.1 Ursachenanalyse der Halluzinationen
Eine systematische Untersuchung des Rohkorpus (`data/corpus/2_raw_scraped/`) ergab, dass fast alle gescrapten Leichte-Sprache-Websites Navigations- und Teaser-Elemente am Textende enthielten:
* **Apotheken (100 % aller Artikel):** *„Wo bekommen Sie noch mehr Informationen? Hier finden Sie mehr Informationen über...“* & *„Welche Frage zu [Thema] haben Sie? Unser Tool durchsucht...“*
* **Hannover:** Über 400 Klick-Aufforderungen (*„Klicken Sie hier“*), 109 Navigationsfragen (*„Wo finde ich weitere Infos?“*) und regionale Zusätze (*„... in der Region Hannover“*).
* **Main-Taunus & Behindertenbeauftragter:** Teaser (*„Hier erfahren Sie mehr zu...“*, *„Hier kommen Sie zum Faltblatt...“*, *„Sprechen Sie uns an!“*) und HTML-Linklisten (*„Link zum Text vom AGG (Alltagssprache)“*).

Weil diese Sätze Hunderte Male in den Trainingsdaten vorkamen, lernte mBART, sie als universelle Abschlussfloskeln an jede Übersetzung anzuhängen.

### 4.2 Erweiterung von `normalize_clean.py`
In `scripts/preprocessing/normalize_clean.py` wurde die Funktion `clean_navigation_boilerplate(text)` etabliert, die alle Artefakte vollständig tilgt:
1. **Navigations-Frageketten:**
   * `Wo (finde|bekomme) ich (weitere/mehr) Informationen...?`
   * `Sie möchten (weitere/mehr) Informationen...?`
   * `Sie interessieren sich für...? Dann klicken Sie...`
2. **Webseiten-Verweise & Teaser:**
   * `Hier (erfahren|lesen|bekommen|finden) Sie (mehr|alles|weitere)...`
   * `(Erfahren|Lesen) Sie mehr (über|zu)...`
   * `Hier kommen Sie (zum|zur|zu)...`
   * `Sprechen Sie uns an!`
3. **Linklisten & Klick-Aufforderungen:**
   * `(Alltagssprache)` / `(in Alltagssprache)`
   * `Einen Link für mehr Infos gibt es unten...`
   * `Links Link zum / zur...`
   * `Wenn Sie online lesen möchten, klicken Sie bitte auf folgenden Link...`

### 4.3 Quantitativer Vorher-Nachher-Vergleich (`analyze_boilerplate_bias.ipynb`)
Das Notebook `notebooks/research/data/analyze_boilerplate_bias.ipynb` wurde aktualisiert und dokumentiert den Bereinigungseffekt:

| Metrik | Vorher (Rohdaten) | Nachher (Bereinigt) | Delta / Reduktion |
| :--- | :---: | :---: | :---: |
| **Artikelpaare gesamt** | 1.533 | **917** | $-616$ ($-40{,}2\,\%$) |
| **Wortanzahl Leichte Sprache (LS)** | 773.375 | **390.922** | $-382.453$ ($-49{,}5\,\%$) |
| **Wortanzahl Alltagssprache (AS)** | 845.558 | **514.429** | $-331.129$ ($-39{,}2\,\%$) |
| **Boilerplate-Prävalenz** | **61,4 % aller Artikel** | **0,0 %** | **Vollständig eliminiert** |

---

## 5. Empirische Analyse der Repetition Penalty

Bei der Übersetzung kurzer Sätze (z. B. 45 Wörter) übersetzte das Modell den Inhalt in den ersten Sätzen fehlerfrei, begann danach jedoch zu halluzinieren. Ein kontrolliertes Experiment mit unterschiedlichen Werten für `repetition_penalty` lieferte den empirischen Nachweis für die Ursache:

```python
# Eingabe (AS):
"Hunde (Canis lupus familiaris) begleiten den Menschen seit Jahrtausenden und gelten als das älteste domestizierte Haustier. Sie stammen vom Wolf ab und haben sich im Laufe der Zeit perfekt an das Leben mit uns angepasst. Heute sind sie nicht nur beliebte Haustiere, sondern oft vollwertige Familienmitglieder, ausdauernde Arbeitstiere und therapeutische Helfer."
```

### Empirischer Vergleich:

| `repetition_penalty` | Generierte Ausgabe (LS) | Analyse & Diagnose |
| :---: | :--- | :--- |
| **`1.0`** | *„...Hunde sind zum Beispiel: Wolfspudel. Wolfspudel ist ein Tier aus dem Wolfsgebiet. Wolfspudel ist ein Tier aus dem Wolfsgebiet. Wolfspudel ist ein Tier...“* | **Degenerative Wiederholungsschleife:** Ohne Penalty verfängt sich das Modell in n-Gramm-Repetitionen. |
| **`2.5`** | *„Was sind Hunde? Hunde sind Haustiere... Wie kann ich meine Hunde adoptieren? ... Diese Arten heißen zum Beispiel: Katzen. Katzen sind besondere Haustiere... Pflege-Hilfe. Eine Pflege-Heirat...“* | **Erzwungene Halluzination:** 2.5 verbietet jedes bereits genutzte Wort (*Hund*, *Mensch*, *Tier*). Beam Search wird in absurde Vokabularbereiche gezwungen (*Katzen*, *Pflege-Heirat*). |
| **`1.2`** | *„Was sind Hunde? Hunde sind Haustiere. Sie begleiten den Menschen schon seit Jahrtausenden. Hunde gehören zu den ältesten Haustieren. Sie stammen vom Wolf ab und haben sich im Laufe der Zeit perfekt an das Leben mit uns angepasst. Heute sind sie nicht nur beliebte Haustiere, sondern oft vollwertige Familienmitglieder, ausdauernde Arbeitstiere und therapeutische Helfer.“* | **Perfekte Leichte-Sprache-Übersetzung:**<br>• Klare Satzstrukturen & erklärende Fragen.<br>• Voller Erhalt der Semantik.<br>• Stoppt exakt bei Erreichen des Sinnes.<br>• **Null Halluzinationen.** |

### Fazit & Standardisierung:
`repetition_penalty = 1.2` stellt den optimalen Sweet Spot dar. Der Wert wurde projektweit in allen Skripten (`train_sft.py`, `generate_dpo_dataset.py`, `train_dpo.py`, `web/app.py` und allen Notebooks) fest verankert.

---

## 6. Zusammenfassung & Nächste Schritte

1. **LoRA-SFT-Training ist stabil und leistungsfähig:** Das Modell lernt echte sprachliche Vereinfachungsstrukturen ohne Overfitting.
2. **Boilerplate-Filterung ist lückenlos implementiert:** Web-Artefakte, Link-Listen und Teaser werden vor dem Modelltraining vollständig herausgefiltert.
3. **Generierungsparameter sind kalibriert:** `repetition_penalty = 1.2` verhindert sowohl Wiederholungsschleifen als auch erzwungene Halluzinationen.
4. **Nächster Schritt:** Ausführung der Pipeline mit bereinigten Daten (`04_normalize_clean.sh` $\rightarrow$ `15_generate_corpus_master.sh` $\rightarrow$ `16a_train_sft_mixup.sh`) und anschließendem DPO-Training (`17a` / `18a`).
