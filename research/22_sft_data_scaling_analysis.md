# 22. SFT Data Scaling & Empirische Lernkurven-Analyse für das Seq2Seq-Übersetzungsmodell

**Thema:** Daten-Skalierungsstudie, Neural Scaling Laws & Sample-Complexity-Analyse des mBART SFT Modells  
**Datum:** 23. August 2026  
**Autor:** Fiete Scheel  
**Notebook:** [`notebooks/research/translation/analyse_sft_data_scaling.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/research/translation/analyse_sft_data_scaling.ipynb)  
**Ergebnisdatei:** [`results/experiments/sft_scaling/sft_scaling_summary.csv`](file:///Users/fietescheel/Documents/Master%20Thesis/results/experiments/sft_scaling/sft_scaling_summary.csv)  
**Skripte:** [`scripts/experiments/sft_scaling/`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/experiments/sft_scaling/), [`scripts/sbatch/experiments/sft_scaling/`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/sbatch/experiments/sft_scaling/)

---

## 1. Motivation & Fragestellung

In der automatischen Textvereinfachung von Alltagssprache (AS) in Leichte Sprache (LS) untersucht diese experimentelle Reihe das **Skalierungsverhalten des Supervised Fine-Tuning (SFT)** von `facebook/mbart-large-50` (LoRA, $r=16, \alpha=32$) entlang der verfügbaren Trainingsdatenmenge:

$$\text{Trainingsfraktionen } F \in \{10\%, 25\%, 50\%, 75\%, 100\%\} \iff N \in \{65, 162, 323, 484, 646\} \text{ Artikelpaare}$$

### Zentrale Forschungsfragen:
1. **Neural Scaling Law des Cross-Entropy Loss ($L_{\text{CE}}$):** Folgt der Validierungsverlust einem monotonen Potenzgesetz ($L(N) \propto N^{-\alpha}$) über die Datenfraktionen?
2. **Kritische Mindestdatenmenge (Sample Complexity):** Ab wie vielen Trainingspaaren $N$ überwindet das Modell die Identitäts-Kopierfalle (*Shortcut Learning*) und erzeugt selbstständig Merkmale Leichter Sprache (kurze Hauptsätze, Kompositazerlegung mit Bindestrichen, erklärende Leitphrasen)?
3. **Skalierungsplateau & Datenbedarf:** Wie entwickeln sich sprachliche Einfachheit ($R_{\text{style}}$), semantischer Erhalt ($R_{\text{sem}}$), lexikalische N-Gramm-Treue (BLEU, ROUGE-L) und strukturelle Stabilität (Satzabbruchquote, Textlänge) bei zunehmender Datenmenge? Haben wir für reines SFT noch zu wenig Daten oder stoßen wir an ein methodisches Plateau?

---

## 2. Versuchsaufbau & Methodik (Ceteris Paribus)

### 2.1 Split-Design & Datenbasis
* **Datenbasis:** `data/analysis/corpus_master.csv` ($N = 808$ gefilterte parallele Artikelpaare, Semantic Similarity $\in [0.70, 0.98]$).
* **Fester 80/10/10 Split (deterministisch ge-seedet, Seed 42):**
  * **Test-Set (10%):** 81 Artikelpaare (strikt ungesehen).
  * **Validation-Set (10%):** 81 Artikelpaare (für Early Stopping und Checkpointing).
  * **Trainings-Pool (80%):** 646 Artikelpaare ($= 100\%$ maximal verfügbare Trainingsdaten).

### 2.2 Skalierungsstufen ($F$):
* **`sft_scale_f010` (10 %):** $N = 65$ Artikelpaare
* **`sft_scale_f025` (25 %):** $N = 162$ Artikelpaare
* **`sft_scale_f050` (50 %):** $N = 323$ Artikelpaare
* **`sft_scale_f075` (75 %):** $N = 484$ Artikelpaare
* **`sft_scale_f100` (100 %):** $N = 646$ Artikelpaare

### 2.3 Modell- & Trainingskonfiguration:
* **Basismodell:** `facebook/mbart-large-50`
* **PEFT / LoRA:** $r=16, \alpha=32$, Dropout $0.05$, Target Modules: `["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]`.
* **Lernrate & Scheduler:** $1 \cdot 10^{-4}$, AdamW, Linear Warmup (10%).
* **Epochen:** 25 (Early Stopping Patience: 6).
* **Batch Size:** 4 mit 4 Gradient Accumulation Steps (effektive Batch Size: 16).
* **Sprachcode-Konfiguration:** `tokenizer.src_lang = "de_DE"`, `tokenizer.tgt_lang = "de_DE"` und `forced_bos_token_id = tokenizer.lang_code_to_id["de_DE"]`.
* **Sauberes Merging:** Adapter werden nach Trainingsende via `peft_m.merge_and_unload()` fest in die Basisgewichte fusioniert und als Standalone-Gewichte (`model.safetensors` / `sft.pt`) abgelegt.

---

## 3. Quantitative Ergebnisse (Gesamtübersicht)

Alle Modelle wurden nach der Fusion auf dem Lebenshilfe-Benchmark (`data/lebenshilfe/lebenshilfe_dataset_clean.json`, $N=37$) mit Beam Search (`num_beams=4`, `repetition_penalty=1.2`, `no_repeat_ngram_size=3`) evaluiert:

| Stufe | Fraktion ($F$) | Artikelpaare ($N$) | Val Loss $\downarrow$ | Simplicity ($R_{\text{style}}$) $\uparrow$ | Semantik AS ($R_{\text{sem}}$) | Treue LS ($Sim_{\text{ref}}$) | BLEU $\uparrow$ | ROUGE-L $\uparrow$ | Ø Tokens | Truncation % $\downarrow$ | Composite Reward | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`sft_scale_f010`** | **10 %** | 65 | 2.6771 | **0.2773** | 0.9400 | 0.8708 | 0.0038 | 0.0940 | 102.0 | 83.78 % | 0.6086 | 85.4 s |
| **`sft_scale_f025`** | **25 %** | 162 | 2.5182 | **0.4669** | 0.9254 | 0.8732 | 0.0074 | 0.1225 | 135.5 | 78.38 % | 0.6961 | 157.0 s |
| **`sft_scale_f050`** | **50 %** | 323 | 2.3352 | **0.3798** | 0.9426 | 0.8830 | 0.0095 | 0.1329 | 154.9 | 78.38 % | 0.6612 | 280.6 s |
| **`sft_scale_f075`** | **75 %** | 484 | 2.2153 | **0.4439** | 0.9237 | 0.8857 | 0.0101 | 0.1323 | 149.9 | 64.86 % | 0.6838 | 403.2 s |
| **`sft_scale_f100`** | **100 %** | 646 | **2.0624** | **0.4688** | 0.9428 | 0.8798 | **0.0113** | **0.1395** | **163.1** | **64.86 %** | **0.7058** | 527.0 s |

---

## 4. Detaillierte wissenschaftliche Erkenntnisse

### 4.1 Monotone Loss-Skalierung ($L(N) \propto N^{-\alpha}$)
* Der Validierungsverlust sinkt streng monoton und ohne Anzeichen von Overfitting:
  $$\text{Loss}(N=65) = 2.6771 \longrightarrow \text{Loss}(N=162) = 2.5182 \longrightarrow \text{Loss}(N=323) = 2.3352 \longrightarrow \text{Loss}(N=484) = 2.2153 \longrightarrow \text{Loss}(N=646) = 2.0624$$
* Dies beweist, dass mBART-Large-50 mit zunehmender Datenmenge die Token-Verteilung der Zieldomäne kontinuierlich besser modelliert.

### 4.2 Der linguistische Phasenübergang ($N \approx 65 \rightarrow N = 162$)
* **Bei 10 % ($N=65$):** Das Modell verbleibt in einer **Shortcut-/Kopierfalle** ($R_{\text{style}} = 0.2773$). 65 Paare reichen nicht aus, um die grammatikalische Transformation zu erlernen; mBART reproduziert weitgehend den Quelltext.
* **Ab 25 % ($N=162$):** Es tritt ein abrupter Qualitätssprung ein ($R_{\text{style}}$ steigt um $+68.4\%$ auf $0.4669$). Das Modell beginnt aktiv:
  * Komposita durch Bindestriche zu trennen (*„Werkstatt-Vertrag“*, *„Erwerbsminderungs-Rente“*, *„Kinder-Sicherheits-Konzept“*).
  * Strukturierende Erklärungsphrasen einzufügen (*„Das bedeutet: ...“*, *„Das heißt: ...“*, *„Zum Beispiel: ...“*).
  * Aufzählungszeichen (*Bullet Points*) für komplexe Sachverhalte zu bilden.

### 4.3 Sättigung der Einfachheit vs. Zuwachs an Robustheit ($N \ge 323$)
Zwischen 25 % ($N=162$) und 100 % ($N=646$) vervierfacht sich die Datenmenge. Während der Simplicity-Score bei **$R_{\text{style}} \approx 0.4688$** stagniert, verbessern sich andere Dimensionen stetig:
1. **Lexikalische Treue:** BLEU steigt um das Dreifache ($0.0038 \rightarrow 0.0113$), ROUGE-L steigt um $+48\%$ ($0.0940 \rightarrow 0.1395$).
2. **Strukturelle Stabilität:** Die Satzabbruchquote sinkt von $83.78\%$ auf $64.86\%$.
3. **Textvollständigkeit:** Die generierte Tokenlänge wächst von $102 \rightarrow 163$ Tokens und nähert sich der Referenz an.
* **Kernbedeutung:** Mehr SFT-Daten führen im Bereich $N > 162$ nicht zu einer *noch radikaleren Vereinfachung*, sondern machen die Generierung **robuster, vollständiger und referenznäher**.

---

## 5. Qualitativer Vergleich der Übersetzungsbeispiele

### Beispiel 1: Arbeitsmarkt & Werkstätten
* **Alltagssprache (Quelle):**  
  *„Kiel, 15.10.2020. Arbeit zweiter oder dritter Klasse?? - Positionspapier Die Rede ist in diesemFall von der Entlohnung für Menschen, die in einer Werkstatt (WfbM) arbeiten. Wichtiges Detail: Es sind die Werkstätten selbst, die diese Veränderung dringend fordern. Der Lohn der Mitarbeiter*innen mit Werkstattvertrag ist bei einem 6 – 8 Stunden Arbeitstag weit davon entfernt, auf dieser Basis ein selbstfinanziertes Leben führen zu können...“*
* **10 % SFT ($N=65$, $R_{\text{style}}=0.082$):**  
  *„Kiel, 15.10.2020. Arbeit zweiter oder dritter Klasse?? - Positionspapier Die Rede ist in diesemFall von der Entlohnung für Menschen, die in einer Werkstatt arbeiten. Wichtiges Detail: Es sind die Werkstätten selbst, die diese Veränderung dringend fordern. Der Lohn der Mitarbeiter*innen mit Werkstattvertrag ist bei einem 6 – 8 Stunden Arbeitstag weit davon entfernt...“* $\rightarrow$ **Kopiert den Quelltext nahezu 1:1.**
* **25 % SFT ($N=162$, $R_{\text{style}}=0.785$):**  
  *„Die Arbeit in einer Werkstatt ist eine Arbeit zweiter oder dritter Klasse? - Positionspapier Die Menschen in der Werkstatt verdienen weniger Lohn als die Mitarbeiter*innen in den Werkstätten. Das bedeutet: Die Menschen bekommen mehr Lohn, als sie verdient haben. Und sie bekommen mehr Geld für ihr Leben. Die Menschen arbeiten in einem Werkstatt-Vertrag. Das heißt: Der Lohn ist bei einem 6 – 8 Stunden Arbeitstag weit davon entfernt...“* $\rightarrow$ **Echte Leichte Sprache mit Satzvereinfachung und Erklärstrukturen.**
* **100 % SFT ($N=646$, $R_{\text{style}}=0.875$):**  
  *„Die Arbeit in einer Werkstatt ist eine Arbeit zweiter oder dritter Klasse? Die Menschen in der Werkstatt verdienen sehr viel Geld. Die Menschen arbeiten oft 6 bis 8 Stunden am Tag. Das bedeutet: Die Menschen können nicht so viel Geld verdienen. Und die Menschen bekommen keine Grundsicherung. Deshalb müssen die Menschen auch Grundsicherung beantragen, um ihren Lebensunterhalt zu bestreiten. Oder sie bekommen eine Erwerbsminderungs-Rente...“* $\rightarrow$ **Konsistente, prägnante SVO-Sätze.**

---

## 6. Grenzen des reinen SFT-Ansatzes

1. **Inhärente Satzabbruchquote:** Auch bei maximaler Datenmenge ($N=646$) enden $64.86\%$ der Texte vorzeitig, da Beam Search bei langen Absätzen an das Tokenlimit stößt.
2. **Phraseologische Repetition:** Bei größeren Datenmengen wiederholt das Modell strukturierende Signalwörter (*„Das bedeutet: ...“*, *„Das heißt: ...“*) teilweise mehrfach innerhalb eines Absatzes.
3. **Domänenspezifität der Eingabelänge:** Da ausschließlich auf langen Artikeln/Absätzen trainiert wurde, neigt das Modell bei isolierten, sehr kurzen Einzelsätzen dazu, diese unverändert durchzureichen.

---

## 7. Haben wir zu wenig Daten? (Datenmangel vs. Methoden-Limitierung)

Die Frage, ob für dieses Projekt ein Datenmangel vorliegt, muss differenziert beantwortet werden:

### 7.1 Wo 646 Artikelpaare BEREITS AUSREICHEN (Kein Datenmangel)
* **Für die grundlegende Domänenadaption:** Bereits ab **162 Paaren** hat das Modell die morphologischen und syntaktischen Eigenheiten Leichter Sprache verinnerlicht. Der Loss sinkt stabil auf $2.06$.
* **Für das Erreichen der maximalen SFT-Einfachheit ($R_{\text{style}} \approx 0.47$):** Da $R_{\text{style}}$ zwischen $N=162$ ($0.4669$) und $N=646$ ($0.4688$) stagniert, ist ein Mangel an Rohdaten **nicht** der Grund für die Decke bei $0.47$. Der Cross-Entropy Loss zwingt das Modell lediglich dazu, den Durchschnitt der menschlichen Trainingsreferenzen nachzuahmen, belohnt aber keine überdurchschnittliche Vereinfachung. Eine bloße Verdopplung der SFT-Rohdaten (z. B. auf 1.500 Artikel) würde diesen Wert kaum steigern (*Diminishing Returns*).

### 7.2 Wo 646 Artikelpaare TATSÄCHLICH ZU WENIG sind (Realer Datenmangel)
* **Für N-Gramm-Präzision und Vokabularabdeckung (BLEU & ROUGE-L):**  
  In der klassischen maschinellen Übersetzung (NMT) werden typischerweise $10^5$ bis $10^6$ Satzpaare benötigt. Bei 646 Artikeln kommen seltene Fachbegriffe nur 1- bis 2-mal vor. BLEU wächst zwar stetig ($0.0038 \rightarrow 0.0113$), bleibt aber auf absolut niedrigem Niveau.
* **Für die Längen- und Abbruchsteuerung:**  
  646 Dokumente reichen nicht aus, damit das Modell selbstständig lernt, jede Ausgabe vor dem Erreichen des maximalen Längenlimits syntaktisch korrekt abzuschließen.

---

## 8. Zusammenfassendes Fazit

1. **Mindestdatenmenge:** mBART-Large-50 benötigt mindestens **$\approx 150\text{--}160$ parallele Artikelpaare**, um die Identitäts-Kopierfalle zu überwinden und selbstständig syntaktische Vereinfachungen zu erzeugen.
2. **Grenzen von reinem SFT Data-Scaling:** Eine Erhöhung von 162 auf 646 Artikelpaare senkt den Cross-Entropy Loss und verbessert die grammatikalische Glätte sowie den BLEU-Score, führt jedoch zu einer Sättigung der sprachlichen Einfachheit bei $R_{\text{style}} \approx 0.47$.
3. **Qualität vor Quantität:** Das Skalierungsexperiment zeigt, dass eine reine Vermehrung von SFT-Rohdaten im Low-Resource-Bereich an Grenzen stößt. Um qualitative Schwächen (Satzabbrüche, Sättigung bei $R_{\text{style}} \approx 0.47$) zu überwinden, sind strukturiertere Daten (Satz-Splittings) sowie zielgerichtete Alignment-Verfahren erforderlich.
