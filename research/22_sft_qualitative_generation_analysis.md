# 22. Qualitative Fehler- und Verhaltensanalyse der SFT-Modellgenerierungen

**Thema:** Qualitative Inhalts- und Fehleranalyse der mBART SFT-Modellausgaben, Faktenstabilität, Korpus-Artefakte und Bias-Untersuchung im Vergleich zur Mixup-Metrik  
**Datum:** 23. August 2026  
**Autor:** Fiete Scheel  
**Datengrundlage:** [`data/temperature_ladder_500/dpo_pairs_w05_w05.jsonl`](file:///Users/fietescheel/Documents/Master%20Thesis/data/temperature_ladder_500/dpo_pairs_w05_w05.jsonl) ($N = 9.030$ bewertete Modellgenerierungen)  
**Bezugsdokumente:** [`research/22_sft_data_scaling_analysis.md`](file:///Users/fietescheel/Documents/Master%20Thesis/research/22_sft_data_scaling_analysis.md), [`research/22_mixup_data_scaling_analysis.md`](file:///Users/fietescheel/Documents/Master%20Thesis/research/22_mixup_data_scaling_analysis.md)

---

## 1. Motivation & Fragestellung

Die quantitative Skalierungsanalyse ([`research/22_sft_data_scaling_analysis.md`](file:///Users/fietescheel/Documents/Master%20Thesis/research/22_sft_data_scaling_analysis.md)) zeigte, dass der Cross-Entropy Validation Loss von mBART zwar kontinuierlich sinkt ($2.677 \rightarrow 2.062$), die sprachliche Einfachheit ($R_{\text{style}}$) jedoch ab $\approx 162$ Trainingspaaren bei einem Plateau von $\approx 0.47$ stagniert. 

Um die **inhärenten Mechanismen und qualitativen Grenzen des reinen Supervised Fine-Tuning (SFT)** aufzudecken, untersucht diese Analyse $9.030$ reale Generierungen aus [`data/temperature_ladder_500/dpo_pairs_w05_w05.jsonl`](file:///Users/fietescheel/Documents/Master%20Thesis/data/temperature_ladder_500/dpo_pairs_w05_w05.jsonl) auf:
1. **Formale Strukturmuster:** Welche Stilelemente der Leichten Sprache wurden erfolgreich erlernt?
2. **Faktentreue & Halluzinationen:** Wie stabil bleiben semantische Relationen, Zahlenkontexte und Entitäten bei der Vereinfachung?
3. **Korpus-Artefakte & Prior-Biase:** Wo memoriert das Modell feste Textbausteine aus Sub-Quellen (z. B. MDR)?
4. **Bias-Transfer auf die Mixup-Metrik:** Besteht das Risiko, dass der in den SFT-Generierungen beobachtete Quell-Bias auf das Mixup-Metrikmodell übergeht?

---

## 2. Qualitative Hauptbefunde aus den Modellgenerierungen

### 2.1 Gelernte Oberflächenstruktur (Erfolgreicher Stiltransfer)
Das SFT-Modell reproduziert die formalen Layout-Konventionen Leichter Sprache mit hoher Zuverlässigkeit:
* **Autonome Leitfragen:** Nahezu jeder generierte Absatz wird mit einer thematischen Einleitungsfrage eröffnet (*„Was passiert mit der Costa Concordia?“*, *„Was ist MOSAIC?“*, *„Was ist Lewis Hamilton?“*, *„Was ist Sofortüberweisung?“*).
* **Morphologische Silbentrennung:** Zusammengesetzte Nomen werden konsequent mit Bindestrichen zerlegt (*„Staats-Anwaltschaft“*, *„Flüchtlings-Betreuung“*, *„Wohn-Container“*, *„Tele-Kommunikations-Markt“*, *„Sport-Wagen“*).
* **Signalwörter & Aufzählungen:** Die Formeln *„Das bedeutet: ...“*, *„Das heißt: ...“* sowie Bullet-Points (*„Zum Beispiel: • ...“*) werden mechanisch in den Textfluss eingebunden.

---

### 2.2 Die 4 qualitativen Fehlermuster des SFT-Modells

Sobald die Generierungen auf inhaltlicher Ebene evaluiert werden, treten fundamentale Schwachstellen zutage:

#### A. Faktenverzerrungen & semantische Zahlen-Fehlinterpretationen (Factual Hallucinations)
mBART verliert bei komplexen syntaktischen Bezügen die Zuordnung von Zahlen zu Entitäten und transformiert Sachverhalte:

| Domäne / Textstelle | Quelltext (Alltagssprache) | SFT-Modellgenerierung | Art des Fehlers |
| :--- | :--- | :--- | :--- |
| **Costa Concordia** *(Zeile 1)* | *„Kapitän Francesco Schettino wurde zu 16 Jahren und einem Monat Haft verurteilt.“* | **„Er wurde im Februar 2015 verhaftet. Und er war 16 Jahre alt.“** | **Haftstrafe wird zum Lebensalter** |
| **Costa Concordia** *(Zeile 1)* | *„Die Costa Concordia hatte vor Giglio einen Felsen gerammt und war gekentert.“* | **„Dort gab es eine schwere Erdbeben-Erschütterung. Das bedeutet: Viele Menschen sind ums Leben gekommen.“** | **Halluzinierte Unglücksursache** |
| **NZZ-Bilanz** *(Zeile 3)* | *„Damit verbuchte sie einen Gewinn von 22,2 Millionen Franken (20,47 Millionen Euro).“* | **„Im Jahr 2047 hat die NZZ Medien-Gruppe einen Gewinn von 22,2 Millionen Euro verdient.“** | **Gewinnsumme wird zur Jahreszahl** |
| **Ferrari Börsengang** *(Zeile 13)* | *„Der ehemalige Formel-1-Weltmeister Ferrari befindet sich auf Talfahrt [...] am Börsenparkett.“* | **„Ferrari ist ein bekannter Sportwagen-Fahrer.“** | **Körperschaft/Marke wird zur Person** |
| **Lewis Hamilton** *(Zeile 12)* | *„Brite könnte schon in Austin seinen dritten WM-Titel fixieren [...] spielt Someone Like You von Adele am Piano.“* | **„Lewis Hamilton ist ein deutscher Formel-1-Pilot [...] Es ist ein Lied von der berühmten Musik-Schreinerin Adele.“** | **Nationalität verfälscht, Halluzination (*„Musik-Schreinerin“*)** |

---

#### B. Memorierte Korpus-Artefakte (*„Sachsen-Anhalt-Präfix“*)
In zahlreichen Generierungen beginnt der Text unvermittelt mit einer geografischen Ortsmarke, selbst wenn der Inhalt völlig andere Länder oder Themen betrifft:
* **Zeile 3 (über die Schweizer NZZ in Zürich):**  
  *„**Sachsen-Anhalt** Die NZZ-Medien-Gruppe hat im vergangenen Jahr mehr Geld verdient...“*
* **Zeile 4 (über den Papst und den Vatikan):**  
  *„**Sachsen-Anhalt** Der Bischof von Probstdorf hat bei einem Treffen mit Franziskus...“*
* **Zeile 6 (über den Telekommunikationsmarkt in Österreich):**  
  *„**Sachsen-Anhalt** Die Telekom-Bürger in Österreich müssen mehr auf Qualität achten...“*

**Ursachenanalyse:**  
Ein substanzieller Anteil der Leichte-Sprache-Trainingsdaten entstammt öffentlich-rechtlichen Landesrundfunkanstalten (MDR Leichte Sprache). Da dort Nachrichtenartikel standardmäßig mit der Ortsmarke *„Sachsen-Anhalt“* eingeleitet werden, hat der autoregressive Decoder von mBART diesen String als dominanten Eröffnungsprior gelernt ($P(y_1 = \text{„Sachsen-Anhalt“}) \gg 0$).

---

#### C. Morphologische Neologismen & Pseudo-Wortschöpfungen
Um dem Druck zur Silbentrennung nachzukommen, erfindet das Modell ungrammatische oder groteske Bindestrich-Wörter:
* *„Bischwürden“* (Zeile 4, statt *Bischofsamt / Bischofswürde*)
* *„Schritt-Früchte“* (Zeile 4, Fehlinterpretation der Redewendung *„den ersten Schritt machen“*)
* *„Kriegs-Kraft-Material“* (Zeile 5, statt *Kriegsmaterial*)
* *„Schiffs-Flatt-Verschwörung“* (Zeile 1)
* *„Preis-Kraft“* (Zeile 6, statt *Preiskampf*)

---

#### D. Längenbudget-Erschöpfung & Vorzeitige Satzabbrüche (Truncation)
In den Modellgenerierungen wird deutlich sichtbar, warum die Truncation Rate selbst bei $100\%$ Datenmenge bei $\approx 65\%$ verharrt:
* Das Modell beginnt am Textanfang mit redundanten Paraphrasen und Leitfragen (*„Was ist ...? Was passiert mit ...?“*).
* Dadurch verbraucht es einen Großteil des Token-Budgets für Einleitungsfloskeln.
* Erreicht die Generierung das maximale Längenlimit (z. B. 256 Tokens), wird die Übersetzung mitten im Nebensatz oder vor der eigentlichen Kernaussage hart abgeschnitten.

---

## 3. Vergleichende Analyse: Generatives SFT vs. Diskriminative Mixup-Metrik

Es stellt sich die methodische Frage, ob der beobachtete Korpus-Bias (*„Sachsen-Anhalt“*) auch das **Mixup-Metrikmodell** (Style Scorer) kontaminiert.

| Eigenschaft | SFT Übersetzungsmodell (mBART-50) | Mixup Metrikmodell (BiLSTM / Transformer Regressor) |
| :--- | :--- | :--- |
| **Modellparadigma** | **Generativ (Autoregressiv)**: Vorhersage von $P(y_t \mid y_{<t}, x)$ | **Diskriminativ (Regressiv)**: Vorhersage eines Skalars $R_{\text{style}} \in [0.0, 1.0]$ |
| **Mechanismus des Bias** | Lernt typische Satzanfänge als bedingte Token-Wahrscheinlichkeit. | Bewertet globale linguistische Dichte- und Repräsentationsmerkmale. |
| **Einfluss von Mixup** | Nicht vorhanden (SFT lernt auf reinen Textpaaren). | **Stochastische Feature-Interpolation**: AS- und LS-Features werden kontinuierlich mit $\lambda \in [0, 1]$ gemischt. |
| **Korrelationsbruch** | Stark an dominante Sequenzmuster gekoppelt. | Einzelne N-Gramme wie *„Sachsen-Anhalt“* verlieren ihre Korrelation zum Label $1.0$, da sie in Tausenden Mixup-Kombinationen auftreten. |
| **Globales Pooling** | Generiert Token für Token (kein Rückbezug auf Textende). | Globales Sequence Pooling (Mean/Max-Pooling über alle Token). Ein einzelnes Orts-Token hat $< 1\%$ Gewicht. |
| **Bias-Risiko** | **Hoch** (Reproduktion von Korpus-Präfixen) | **Nahezu 0 %** (Resistent gegen einzelne Vokabular-Artefakte) |

---

## 4. Wissenschaftliche Schlussfolgerungen für die Masterarbeit

1. **Formale Entkopplung von Form und Inhalt im SFT:**  
   Reines Supervised Fine-Tuning optimiert die *statistische Token-Überlappung* mit Trainingsreferenzen. Das Modell meistert dadurch die formale Syntax Leichter Sprache (Bindestriche, kurze Sätze, Einleitungsfragen), versagt jedoch bei der *semantischen Konsistenz* (Zahlen- und Faktenlogik).
2. **Grenzen quantitativer SFT-Vermehrung:**  
   Eine Erhöhung der SFT-Datenmenge führt dazu, dass stereotype Floskeln (*„Das bedeutet: ...“*, *„Sachsen-Anhalt ...“*) noch fester eingeprägt werden, ohne das logische Textverständnis zu verbessern.
3. **Notwendigkeit zielgerichteter Alignment-Verfahren:**  
   Um die identifizierten Hauptprobleme (Faktenverzerrungen, Neologismen, Satzabbrüche) zu überwinden, sind differenziertere Mechanismen erforderlich:
   * **Preference Optimization (DPO):** Bestraft Faktenhalluzinationen und belohnt semantisch treue, kompakte Vereinfachungen.
   * **Sentence-Level Granularität:** Zerlegung von Dokumenten in Einzelsatz-Paare, um Längenüberschreitungen und Ausfransungen am Textende zu eliminieren.
   * **Robuste Metrik-Führung:** Das Mixup-Modell bietet durch seine diskriminative Architektur eine bias-freie Bewertungsbasis für das nachfolgende Alignment.
