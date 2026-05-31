# 09 Dataset Analysis: Fehlerkategorien & Fallbeispiele

Dieses Dokument dokumentiert spezifische Problemfälle und **Grenzfälle** im Korpus, die durch die semantische Ähnlichkeitsanalyse identifiziert wurden. Diese Beispiele dienen als Grundlage für die Verfeinerung der Filterstrategien.

## 1. Kategorie: Technisches Rauschen / Teaser-Listen
**Problem:** Eine AS-Seite mit vielen Inhalten (z. B. eine Presseliste) wird mit einem statischen LS-Satz oder einem Teaser verglichen.

### ID: 55 | Source: behindertenbeauftragter | Sim: 0.2231
- **AS URL:** https://www.behindertenbeauftragter.de/DE/AS/presse-und-aktuelles/presse-und-aktuelles-node.html
- **LS URL:** https://www.behindertenbeauftragter.de/DE/LS/presse-und-aktuelles/presse-und-aktuelles.html
- **Fehler:** Die AS enthält eine lange Liste von Pressemitteilungen. Die LS enthält nur den statischen Einleitungssatz: *"Hier finden Sie Informationen für die Presse. Und Publikationen und Erklärungen."*
- **Fazit:** Kein Trainingswert.

---

## 2. Kategorie: Platzhalter-Texte (Lorem Ipsum)
**Problem:** Die LS-Version enthält noch Testdaten oder Platzhalter, die beim Scraping nicht als solche erkannt wurden.

### ID: 33 | Source: behindertenbeauftragter | Sim: 0.4061
- **AS URL:** https://www.behindertenbeauftragter.de/DE/AS/presse-und-aktuelles/veranstaltungen/veranstaltungen-node.html
- **LS URL:** https://www.behindertenbeauftragter.de/DE/LS/presse-und-aktuelles/veranstaltungen/veranstaltungen.html
- **Fehler:** Der LS-Text besteht fast vollständig aus *"Lorem ipsum dolor sit amet..."*.
- **Fazit:** Muss zwingend gefiltert werden.

---

## 3. Kategorie: Themen-Shift / Schlechtes Alignment
**Problem:** Die URLs sind zwar ähnlich, die Inhalte behandeln aber unterschiedliche Unterthemen derselben Kategorie.

### ID: 149 | Source: hamburg | Sim: 0.5235
- **AS URL:** https://www.hamburg.de/freizeit/ausfluege/in-hamburg/ausfluege-in-hamburg-368184
- **LS URL:** https://www.hamburg.de/barrierefrei/leichte-sprache/freizeit/ausfluege-575864
- **Fehler:** Während die AS über den **Elbstrand** schreibt, listet die LS Ausflugsziele wie das **Freilicht-Museum Hitzacker** und den **Serengeti Park** auf.
- **Fazit:** Die semantische Ähnlichkeit von 0.52 erkennt korrekt, dass beide "Freizeit" behandeln, aber nicht denselben Ort.

---

## 4. Kategorie: Identische Texte (Keine Vereinfachung)
**Problem:** AS und LS sind exakt gleich. Oft handelt es sich um Menüstrukturen oder Seiten, die bereits in LS vorlagen, aber in beiden Verzeichnissen identisch ausgespielt werden.

### ID: 131 | Source: stuttgart | Sim: 1.0000
- **AS URL:** https://www.stuttgart.de/leichte-sprache-index
- **LS URL:** https://www.stuttgart.de/leichte-sprache-index?sp%3Aout=easy
- **Fehler:** Beides sind identische Linklisten zu weiteren LS-Artikeln. 
- **Fazit:** Wertlos für das Lernen einer Übersetzung.

### ID: 206 | Source: koeln | Sim: 1.0000
- **AS URL:** .../wohnsitz-abmelden (AS)
- **LS URL:** .../wohnsitz-abmelden (LS)
- **Fehler:** In beiden Fällen wurde bereits die LS-Version ("Information in Leichter Sprache...") extrahiert.
- **Fazit:** Duplikate entfernen.

---

## 5. Kategorie: "Fast" identisch (Grenzfall)
**Problem:** Die Texte sind inhaltlich sehr nah beieinander, aber die "Alltagssprache" ist bereits sehr einfach oder die LS ist nur eine minimale Modifikation.

### ID: 46 | Source: behindertenbeauftragter | Sim: 0.9808
- **AS URL:** .../20220321_Welt_DS_Tag.html
- **LS URL:** .../20220321_Welt_DS_Tag_LS.html
- **Analyse:** Die AS ist hier bereits in einer Art "Einfachen Sprache" verfasst. Die LS ändert nur Details (z.B. "Welt-Down-Syndrom Tag" statt "Welt-Down-Syndrom-Tag").
- **Fazit:** Solche Fälle blähen das Korpus auf, ohne dem Modell komplexe Vereinfachungen beizubringen.

---

## 6. Kategorie: Grenzfälle Untergrenze (~0.60)
**Frage:** Sind Texte bei 0.60 noch als "Übersetzung" brauchbar oder bereits zu weit weg?

### Fall A: MDR (Sim: 0.6011) - Teil-Übersetzung
- **AS:** Brand am Hasselbachplatz in Magdeburg (Detailbericht).
- **LS:** Feuer an Silvester in Magdeburg **und** Erfurt.
- **Problem:** Die LS fasst zwei verschiedene Nachrichten zusammen, während die AS nur eine behandelt. 
- **Urteil:** Bedingt brauchbar, aber führt "Rauschen" (Erfurt-News) ein.

### Fall B: brandeins (Sim: 0.6047) - Literarische Freiheit
- **AS:** Hochgradig literarischer Text über die Jagd ("beschlagene Ricke", "Träger waagerecht").
- **LS:** Radikale Vereinfachung ("Ich darf das Reh erschießen. Darüber freue ich mich.").
- **Urteil:** **Extrem wertvoll.** Genau das ist die Aufgabe von LS: komplexe Metaphern in Fakten zu übersetzen. Ein Filter bei 0.6 würde diesen "Gold-Standard" fast löschen.

### Fall C: Wiesbaden (Sim: 0.6120) - Generalisierung
- **AS:** "Filmstadt Wiesbaden" (FSK, Murnau-Stiftung, Filmerbe).
- **LS:** "Caligari Film Bühne" (Ein spezifisches Kino).
- **Urteil:** Zu spezifisch. Die LS ist keine Übersetzung des AS-Textes, sondern ein verwandtes Thema.

---

## 7. Kategorie: Grenzfälle Obergrenze (~0.96 - 0.97)
**Frage:** Ist 0.98 zu streng oder zu locker als Obergrenze?

### Fall D: Stuttgart (Sim: 0.9719) - Perfekte LS
- **AS:** Einleitung zum Generationenhaus.
- **LS:** Fast identische Struktur, aber Begriffe wie "Generationen-Häuser" (mit Bindestrich) und kürzere Sätze.
- **Urteil:** **Behalten.** Das ist eine hochwertige, strukturerhaltende Übersetzung.

### Fall E: Salon im Kleisthaus (Sim: 0.9675) - Stilistisches Polishing
- **AS:** Digitales Talkformat, Zeitenwende 2020.
- **LS:** Glättet den Text, macht ihn direkter.
- **Urteil:** **Behalten.** Auch wenn die Ähnlichkeit hoch ist, zeigt es die feinen Nuancen der Simplifizierung.

---

## Finale Empfehlung zur Filterung (Revidiert)

Nach Sichtung der Grenzfälle empfiehlt sich eine leicht angepasste Strategie:

1.  **Untergrenze:** Bleibt bei **0.60**. Texte darunter (wie ID 149 mit 0.52) sind oft Themen-Shifts. Texte bei 0.60 (Jagd-Beispiel) sind "High-Level" Simplifizierungen, die wir brauchen.
2.  **Obergrenze:** Erhöhung auf **0.99**. Bei 0.97 (Stuttgart) finden wir noch sehr gute, echte Übersetzungen. Erst über 0.99 handelt es sich fast immer um exakte Kopien ohne Mehrwert.
3.  **Zusatz-Filter:**
    *   **Länge:** LS-Texte mit weniger als 10 Tokens (Teaser-Leichen wie ID 82) sollten unabhängig von der Similarity fliegen.
    *   **Lorem Ipsum:** Harte Suche nach "Lorem ipsum" Texten.
