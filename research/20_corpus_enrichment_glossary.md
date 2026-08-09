# Korpus-Anreicherung durch Glossar-Erklärungen (Woche 20)

Dieses Dokument beschreibt die Pipeline zur automatischen Identifikation schwieriger Begriffe und deren Anreicherung mit Definitionen und Begriffserklärungen aus dem **Hurraki-Glossar**.

---

## 1. Motivation
Im Regelwerk der Leichten Sprache (LS) ist festgelegt, dass Fremdwörter, Fachbegriffe und komplexe Zusammenhänge nicht nur vereinfacht, sondern bei ihrer Verwendung auch explizit erklärt werden müssen (z. B. durch Appositionen oder angehängte Glossareinträge). 

Um diese Strukturierung maschinell zu erlernen, reicht ein Standard-Satzalignment oft nicht aus. Durch eine datenseitige Augmentierung lernt das Übersetzungsmodell, schwierige Wörter im Alltagssprache-Text (AS) automatisch zu erkennen und die zugehörigen Erklärungen in LS-Übersetzungen zu integrieren.

---

## 2. Die Pipeline zur Glossar-Anreicherung

Die Implementierung ist in zwei separate Vorverarbeitungs-Skripte aufgeteilt:

### Schritt A: Aufbau des Glossars ([`3_build_glossary.py`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/preprocessing/3_build_glossary.py))
Extrahiert Definitionen und Begriffserklärungen aus Online-Lexika für Leichte Sprache (z. B. Hurraki) und überführt diese in ein strukturiertes JSON-Format unter [`data/vocabs/hurraki_glossary.json`](file:///Users/fietescheel/Documents/Master%20Thesis/data/vocabs/hurraki_glossary.json).

### Schritt B: Korpus-Augmentierung ([`4_enrich_glossary.py`](file:///Users/fietescheel/Documents/Master%20Thesis/scripts/preprocessing/4_enrich_glossary.py))
Das Skript lädt den bereinigten Korpus sowie das Glossar und führt folgende Schritte durch:

1. **Wortextraktion:** Der Alltagstext (`as_text`) wird in bereinigte Kleinbuchstaben-Tokens zerlegt.
2. **Abgleich:** Es wird geprüft, welche Wörter im Hurraki-Glossar vorkommen.
3. **Erklärungs-Suffix:** Für alle gefundenen Wörter wird ein Erklärungsblock im Format:
   `[Wort]. [Wort] bedeutet: [Definition]` generiert.
4. **Paar-Duplizierung und Augmentierung:** 
   * Das Original-Paar bleibt erhalten (um einfaches Übersetzen ohne Begriffserklärungen zu stützen).
   * Es wird ein neues Paar hinzugefügt, bei dem die Erklärungen als Suffix an das `ls_text`-Feld angehängt werden:
     $$\text{ls\_text}_{\text{augmented}} = \text{ls\_text}_{\text{original}} + \text{"\textbackslash n\textbackslash n"} + \text{Erklärungen}$$

---

## 3. Auswirkungen auf das Training

Modelle, die auf diesem angereicherten Datensatz (`enriched`) trainiert wurden (z. B. `3_dpo_w05_w05_enriched`), zeigen im Vergleich zu den Basismodellen folgende qualitative Verbesserungen:

* **Erhöhte Informationsdichte:** Die Übersetzung liefert dem Endnutzer die notwendigen Hilfestellungen direkt im Text.
* **Besseres semantisches Alignment:** Durch das Einfügen der Begriffserklärungen steigt die semantische Ähnlichkeit zur echten Leichten-Sprache-Referenz (von **0.8334** auf **0.8551**) sowie der SBERT-Score zur Quelle (von **0.8733** auf **0.9058**), da Erklärungen dem Informationsverlust entgegenwirken.
