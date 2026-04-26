# Datensatz-Erstellung: Aufbau des Text-Korpus

Dieses Dokument dokumentiert die Erstellung des Text-Korpus für die Masterarbeit. Es beschreibt den Prozess von der initialen Extraktion bis hin zur qualitativen Verbesserung der Daten für die einzelnen Quellen.

## Methodik

Der Aufbau des Korpus erfolgte in zwei Hauptphasen:
1.  **Alignment-Phase:** Identifizierung von URL-Paaren (Leichte Sprache <-> Alltagssprache) und Speicherung in `results/aligned_urls/`.
2.  **Extraktions-Phase:** Gezielter Download der Inhalte basierend auf den URL-Paaren und Speicherung der bereinigten Texte in `results/corpus/`.

---

## 1. Apotheken Umschau

### Initialer Status
*   **Skript:** `apotheken_scraper.py`
*   **Vorgehen:** Basierte auf der Suche nach Links innerhalb der `.article-body` oder `<article>` Tags.
*   **Probleme:** 
    *   Extraktion von Copyright-Hinweisen (z.B. "© W&B/...") mitten im Text.
    *   Bildbeschreibungen ("Das Bild zeigt...") wurden als Fließtext erfasst.
    *   Inhaltsverzeichnisse am Anfang der AS-Artikel störten den Textfluss.
    *   Aufforderungen zur Registrierung ("Jetzt kostenlos anmelden") waren enthalten.

### Verbesserungen
*   **Decomposition:** Gezieltes Löschen von `<figcaption>`, `<figure>`, sowie Elementen mit Klassen wie `copyright`, `teaser` oder `related-articles` mittels BeautifulSoup `decompose()`.
*   **TOC-Filter:** Automatisches Entfernen von Listen (`<ul>`), die überwiegend aus internen Anker-Links (`#`) bestehen.
*   **Boilerplate-Filter:** Implementierung einer Blacklist für Sätze, die Copyright-Symbole enthalten oder mit "Das Bild zeigt" / "Die Grafik zeigt" beginnen.
*   **Interaktions-Filter:** Aggressives Filtern von Werbe- und Anmeldeaufforderungen.

---

## 2. Bundes-Behindertenbeauftragter

### Initialer Status
*   **Skript:** `behindertenbeauftragter_scraper.py`
*   **Vorgehen:** Suche im `<div id="content">`.
*   **Probleme:**
    *   **Fallback-Links:** Viele Alltagssprache-Links führten aufgrund von Webseiten-Umstrukturierungen nur auf die Startseite.
    *   **PDF-Seiten:** In Leichter Sprache gab es Seiten, die keinen Text enthielten, sondern nur einen Download-Link zu einem PDF.
    *   **Menü-Reste:** "Top Meldung", "Weitere Themen" oder "Dokument vorlesen" wurden mit extrahiert.

### Verbesserungen
*   **URL-Validierung:** Paare werden nun ignoriert, wenn die AS-URL auf `startseite-node.html` verweist.
*   **Inhalts-Validierung:** LS-Seiten mit weniger als 50 Tokens, die Schlüsselwörter wie "Herunterladen" enthalten, werden als reine Download-Seiten verworfen.
*   **Strukturelle Reinigung:** Entfernung von CSS-Klassen wie `c-teaser`, `c-nav`, `c-audio-player` und `c-download-box`.
*   **Ergebnis:** Die Anzahl der Paare sank von 73 auf 60, dafür stieg die Textqualität massiv an.

---

## 3. Brand eins

### Initialer Status
*   **Skript:** `brandeins_scraper.py`
*   **Vorgehen:** Extraktion beider Sprachstufen von einer einzigen URL. Da AS und LS oft im selben Textblock stehen, basierte die Trennung initial auf einer simplen Heuristik (erster Paragraph = AS, Rest = LS).
*   **Probleme:** 
    *   **Parsing-Fehler:** Die strukturelle Heuristik schlug oft fehl, wodurch LS-Texte im AS-Feld landeten und umgekehrt.
    *   **Boilerplate:** Jeder Artikel enthielt den gleichen Einleitungssatz ("Die Leichte Sprache nimmt den Inhalt ernst...") sowie Autorenzeilen ("Text: Holger Fröhlich").
    *   **Unsichtbare Formatierung:** Die farbliche Kennzeichnung (Rot für LS) war im HTML oft in verschachtelten `<span>`-Tags versteckt, die vom Scraper ignoriert wurden.

### Verbesserungen
*   **Deep-Color-Inspection:** Der Scraper prüft nun das gesamte HTML jedes Absatzes auf rote Farbcodes (`#ff0000`, `#fa4600` etc.) und `<strong>`-Tags. Dies ermöglicht eine präzise Trennung, auch wenn die Struktur innerhalb der Absätze variiert.
*   **Aggressives Cleaning:** Systematische Entfernung des Standard-Einleitungssatzes und variierender Vorspann-Phrasen ("Hier die Übersetzung von...") per Regex.
*   **Autoren-Filter:** Automatisches Ausfiltern von Namenszeilen und Credit-Fragmenten.
*   **Ergebnis:** Ein hochgradig balanciertes Korpus (ca. 167 LS-Tokens vs. 189 AS-Tokens im Durchschnitt) ohne strukturelle Vermischung.

---

## 4. Hamburg.de

### Initialer Status
*   **Skript:** `hamburg_scraper.py`
*   **Vorgehen:** Suche nach Links zum "Originaltext" mittels CSS-Selektoren.
*   **Probleme:** 
    *   **Fehl-Alignments:** Die alte Logik nutzte zu schwache Selektoren und griff bei fehlendem Sprachumschalter auf beliebige Links im Fließtext zurück. Dies führte dazu, dass dutzende LS-Artikel fälschlicherweise derselben AS-URL (z.B. "Grundsteuer" oder "Starkregen") zugeordnet wurden.
    *   **Maschinelle Übersetzung:** Ein signifikanter Teil der LS-Texte auf Hamburg.de wurde automatisch generiert (Hinweis: "Ein Computer hat diesen Text übertragen"). Diese Texte entsprechen oft nicht den Qualitätsstandards für Leichte Sprache.
    *   **Boilerplate-Noise:** AS-Texte enthielten oft Banner-Hinweise auf automatische Übersetzungen. LS-Texte enthielten am Ende oft mehrfache Impressum-Blöcke ("Büro für Leichte Sprache Köln...").

### Verbesserungen
*   **Sprachleisten-Fokus:** Die Suche nach dem AS-Gegenstück wurde strikt auf die offizielle Sprachleiste (`.km1-language-bar`) begrenzt. Fallback-Suchen im restlichen Dokument wurden entfernt, um "Geister-Alignments" zu verhindern.
*   **MT-Filter:** Texte, die den Disclaimer für maschinelle Übersetzung enthalten, werden nun systematisch identifiziert und komplett aus dem Datensatz entfernt.
*   **Surgical Cleaning:** Gezielte Filterung von Standard-Bannern ("Bitte beachten Sie, dass dieser Inhalt...") und Übersetzer-Nennungen per Regex während der Extraktion.
*   **Ergebnis:** Reduktion von 155 auf 57 qualitativ hochwertige, manuell geprüfte Paare mit präzisem Themen-Alignment.

---

## 5. Stadt Köln

### Initialer Status
*   **Skript:** `koeln_scraper.py`
*   **Vorgehen:** Extraktion von Inhalten aus dem Wayback Machine Archiv, da viele historische LS-Texte nicht mehr direkt live verfügbar sind.
*   **Probleme:** 
    *   **Encoding-Fehler:** Massive Probleme mit Umlauten (z. B. `kÃ¶nnen`), da die automatische Erkennung von `requests` bei den Archiv-Seiten oft fehlschlug.
    *   **Text-Duplikation:** Die Struktur der Kölner Service-Seiten führte bei einfacher Tag-Extraktion (`p`, `li`) zu doppelten Inhalten, wenn Paragraph-Tags innerhalb von Listen-Elementen verschachtelt waren.
    *   **Boilerplate-Noise:** AS-Texte enthielten umfangreiche Formularreste ("War dieser Artikel hilfreich?", "Ihre E-Mail-Adresse"), Kontaktboxen und Vorlese-Funktionen.
    *   **Hub-Pages:** Einige AS-URLs verwiesen auf reine Verteilerseiten mit minimalem Textanteil, was zu einem starken Ungleichgewicht gegenüber den LS-Texten führte.

### Verbesserungen
*   **Encoding-Autodetect:** Umstellung auf `response.apparent_encoding`, um die korrekte Zeichenkodierung der archivierten HTML-Snapshots sicherzustellen.
*   **Smart Content Selection:** Implementierung eines Parent-Checks während der Extraktion, um verschachtelte Duplikate systematisch zu überspringen.
*   **String-Deduplizierung:** Zusätzlicher Filter für identische Textblöcke, um redundante Inhaltsabschnitte innerhalb eines Artikels zu eliminieren.
*   **Aggressiver Boilerplate-Filter:** Erweiterung der Blacklist um spezifische Phrasen der Kölner Feedback- und Kontaktformulare.
*   **Robustes Prozess-Management:** Einführung eines inkrementellen Speichersystems, um den Fortschritt auch bei den häufigen Verbindungsabbrüchen zum Wayback-Archiv zu sichern.
*   **Ergebnis:** Ein hochqualitatives Teilkorpus (aktuell 39 Paare) mit sauberen Umlauten und minimiertem Rauschen.

---

## 6. Lebenshilfe Main-Taunus

### Initialer Status
*   **Skript:** `main_taunus_scraper.py`
*   **Vorgehen:** Extraktion von Inhalten aus dem Wayback Machine Archiv der Lebenshilfe Main-Taunus.
*   **Probleme:** 
    *   **Platzhalter-Inhalte:** Viele Seiten enthielten nur den Hinweis "Bald steht hier ein Text in Leichter Sprache" oder "Bitte um etwas Geduld", was wertlose Datensätze erzeugte.
    *   **Content-Duplikate:** Die Webseite nutzt verschiedene Navigations-IDs in der URL (z.B. `m-20`, `m-79`), die auf den identischen Inhalt führen. Dies führte zu massiven Redundanzen im Korpus.
    *   **Technisches Rauschen:** Fragmente wie `(Diese Datei existiert leider nicht mehr.)` oder `mutex/ocfipreoqyfb/mutex` wurden mitextrahiert.
    *   **Linguistischer "Ballast":** Fast jeder Artikel endete mit umfangreichen Kontaktblöcken (Telefonnummern, E-Mails, Adressen, Spendenkonten), die linguistisch repetitiv sind und das LS-AS-Verhältnis verfälschen.
    *   **Informationsarme Seiten:** Reine Namenslisten (z.B. der Künstlergalerie) wurden als Fließtext erfasst.

### Verbesserungen
*   **Platzhalter-Ausschluss:** Implementierung eines Filters, der Texte mit Phrasen wie "Bald steht hier..." oder "Geduld" systematisch erkennt und das gesamte Paar verwirft.
*   **Content-Hashing:** Einführung einer De-Duplizierung basierend auf dem Hash des bereinigten Textinhalts. Identische Texte werden unabhängig von ihrer URL nur einmal aufgenommen.
*   **Surgical Cleanup (Regex):** Entfernung von technischen Artefakten und Navigations-Phrasen ("Hier kommen Sie zum Faltblatt", "Hier erfahren Sie mehr") mittels regulärer Ausdrücke.
*   **Kontakt-Truncation:** Automatisches Abschneiden der Texte bei Signalwörtern wie "Ansprechpartner", "Kontakt:" oder "Adresse:". Dies stellt sicher, dass nur der eigentliche redaktionelle Inhalt im Korpus verbleibt.
*   **Heuristischer Token-Filter:** Paare mit weniger als 20 Tokens nach der Bereinigung werden als unzureichend verworfen.
*   **Encoding-Optimierung:** Nutzung von `response.content` (Bytes) für BeautifulSoup, um die im HTML deklarierte Zeichenkodierung des Archivs korrekt zu interpretieren.
*   **Ergebnis:** Eine signifikante Reduktion des Rauschens; der Korpus besteht nun aus 36 hochwertigen, distinkten Textpaaren.

---

## 7. MDR (Mitteldeutscher Rundfunk)

### Initialer Status
*   **Skript:** `mdr_scraper.py`
*   **Vorgehen:** Extraktion von News-Beiträgen und deren Archiv-Gegenstücken in Alltagssprache.
*   **Probleme:** 
    *   **Massive Textduplikation ("Echo-Effekt"):** Fast jeder Absatz wurde doppelt extrahiert, da der Selektor (`div.paragraph, p.text`) sowohl den Container als auch das darin liegende Paragraph-Tag erfasste.
    *   **Strukturelles Rauschen:** Metadaten wie "Hauptinhalt", "Neuer Bereich", "Neuer Abschnitt" und Bildrechte wurden als Fließtext mitgenommen.
    *   **Boilerplate-Anhänge:** LS-Artikel endeten oft mit Verweisen auf Radio-Studios oder Sendungen ("Über dieses Thema berichtet der MDR auch in schwerer Sprache...").
    *   **Extreme Längenunterschiede:** Live-Ticker oder Sammelartikel in Alltagssprache (oft > 3000 Tokens) wurden kurzen LS-Meldungen (ca. 200 Tokens) zugeordnet, was für das Alignment unbrauchbar war.

### Verbesserungen
*   **Nested-Element-Filter:** Implementierung eines Parent-Checks (`any(parent in candidates for parent in el.parents)`), der sicherstellt, dass Text innerhalb verschachtelter Tags nur einmal (vom obersten relevanten Element) extrahiert wird.
*   **Targeted Content Area:** Begrenzung der Suche auf den `<main>` oder `<article>` Bereich, um globale Navigationselemente von vornherein auszuschließen.
*   **Smarte Skip-Signale:** Einführung einer Blacklist für strukturelle MDR-Begriffe (z. B. "Hauptinhalt", "Neuer Bereich") und Metadaten, die während der Iteration über die HTML-Elemente gefiltert werden.
*   **Ratio-Filter:** Einführung eines Grenzwerts für das Längenverhältnis. Textpaare mit einer Ratio > 5.0 (AS massiv länger als LS) werden automatisch aussortiert, um Ticker und Megaseiten zu eliminieren.
*   **Bulletpoint-Optimierung:** Automatische Isolation von Aufzählungszeichen (`•`) durch Leerzeichen für eine saubere Tokenisierung.
*   **Ergebnis:** Ein hocheffizienter Korpus mit 235 validen Paaren. Die durchschnittliche Token-Länge sank durch die Duplikat-Entfernung auf realistische Werte (ca. 239 LS-Tokens vs. 354 AS-Tokens).

---

## 8. Sozialpolitik.com

### Initialer Status
*   **Skript:** `sozialpolitik_scraper.py`
*   **Vorgehen:** Suche nach "Standardsprache"-Links in LS-Artikeln.
*   **Probleme:** 
    *   **Identische Texte:** Bei vielen Paaren (ca. 30%) waren LS und AS Text zu 100% identisch, da das CMS für beide Versionen denselben Inhalt auslieferte (besonders bei News/Wettbewerben).
    *   **Boilerplate-Code:** Mitextraktion von interaktiven Elementen wie Quizzen ("Teste dein Wissen"), Download-Bereichen ("Hol dir das Magazin") und umfangreichen Seitenleisten (`aside`).
    *   **Strukturverlust:** Fließtext wurde ohne Zeilenumbrüche zusammengefügt, was Listen und Absätze unkenntlich machte.
    *   **Fehl-Alignments:** Die Startseite (`/es/`) und Übersichtsseiten wurden fälschlicherweise als Artikel-Paare erfasst.

### Verbesserungen
*   **Identitäts-Filter:** Automatischer Vergleich von LS- und AS-Texten. Identische Paare werden nun konsequent aussortiert, um die Trennschärfe des Korpus zu wahren.
*   **Surgical Decomposing:** Explizites Entfernen von CSS-Klassen wie `.quiz-container`, `.download-area` und dem `<aside>`-Tag vor der Textextraktion.
*   **Strukturerhalt:** Block-Elemente (`p`, `h`, `li`) werden nun mit Zeilenumbrüchen (`\n`) zusammengefügt, um die logische Gliederung des Textes für das Satz-Alignment zu erhalten.
*   **Root-Filter:** Expliziter Ausschluss der Startseiten-URLs (`/es/`) aus der Alignment-Liste.
*   **Ergebnis:** Ein Korpus von 15 hochqualitativen Paaren mit einer deutlichen Längendifferenz (Durchschnitt: 345 LS-Tokens vs. 821 AS-Tokens).

---

## 9. taz (Die Tageszeitung)

### Initialer Status
*   **Skript:** `taz_scraper.py` (sowohl für URL-Alignment als auch Content-Extraktion)
*   **Vorgehen:** Suche nach Links mit den Wörtern "schweren" und "text" in den Leichte-Sprache-Artikeln, um die alltagssprachlichen Pendants zu finden.
*   **Probleme:**
    *   **Fehlerhaftes Alignment (Zeit-Diskrepanz):** Der Scraper unterschied nicht zwischen Links zu spezifischen Quellartikeln und Links zu Sammel-/Themenseiten (z. B. "Texte zur Europawahl"). Dies führte dazu, dass LS-Texte von 2019 mit AS-Übersichtsseiten von 2024 gepaart wurden.
    *   **Metadaten und Boilerplate:** LS-Texte begannen fast immer mit dem Hinweis "Hier können Sie den Text herunterladen" oder endeten mit Verweisen auf den "schweren Text".
    *   **Spendenaufruf in AS-Texten:** Die AS-Artikel der *taz* schließen standardmäßig mit einem langen Spendenaufruf ("Als Genossenschaft gehören wir unseren Leser:innen..."). Da dieser die CSS-Klasse `.article` trug, wurde er stets als redaktioneller Inhalt mitextrahiert (ca. 100 Token Rauschen pro Artikel).
    *   **1:N-Mapping (Aggregation):** Ein LS-Text der *taz* fasst häufig die Kernaussagen von bis zu drei separaten AS-Artikeln zusammen.

### Verbesserungen
*   **Targeted URL-Filtering:** Implementierung eines Filters im Alignment-Skript, der Tag- und Übersichtsseiten (erkennbar an `/!t` oder `/!p` in der URL) explizit ignoriert, um nur direkte Artikel-Links zuzulassen.
*   **Boilerplate-Decomposition:** Gezieltes Löschen des Spendenaufrufs über die Entfernung der HTML-Container `.tzi-bottom-container`, `.tziBottom` und `.social-media-title` vor der Textextraktion.
*   **String-Matching-Filter:** Implementierung spezifischer String-Filter, um Standardphrasen ("Hier können Sie den Text herunterladen", "Dieser Text ist Werbung", "──────────────────") aus dem Fließtext zu tilgen.
*   **Ergebnis:** Das Rauschen wurde drastisch reduziert (Entfernung von ca. 1.400 unnötigen AS-Token und 800 LS-Token über das Test-Set). Das Token-Verhältnis blieb stabil bei ca. 1.85, die inhaltliche Qualität und Kohärenz der Paare stieg jedoch massiv an. Das inhärente Problem der 1:N-Aggregation bleibt als Charakteristikum der *taz*-Texte bestehen und muss bei feingranularem Alignment berücksichtigt werden.

---

## 10. Architektur der Korpus-Scraper (Allgemein)
