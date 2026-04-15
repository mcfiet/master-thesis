# Übersicht der Scraping-Quellen und Alignment-Strategien (Leichte Sprache <-> Alltagssprache)

Dieses Dokument dient als "Living Document" (fortlaufend aktualisiert) für die Entwicklung von Web-Scrapern zur Erstellung eines parallelen Textkorpus (Standarddeutsch / Alltagssprache vs. Leichte/Einfache Sprache). Es aggregiert die Erkenntnisse aus unserer Recherche, dem Toborek-Paper (2023) und der analysierten Excel-Quellenliste.

## 1. Explizite Link-Verknüpfung (Direkte Backlinks)

Diese Quellen sind am verlässlichsten, da die Version in Leichter Sprache (LS) direkt auf die Version in Alltagssprache (AS) verlinkt (oder umgekehrt). Hier kann das Alignment 1:1 durch das Verfolgen von Links automatisiert werden.

### MDR (Mitteldeutscher Rundfunk)
*   **Status:** `Sehr gut geeignet`
*   **Strategie (Toborek / Eigene Analyse):** Suche im LS-Artikel nach einem Teaser-Block für die "schwere" Version. Spezifisch nach einem Element mit der Klasse `conHeadline` und dem Text `"Hier können Sie diese Nachricht auch in schwerer Sprache lesen:"`. Der darin liegende Link führt zum AS-Artikel.
*   **Beispielpaar:**
    *   **AS:** [Prozess Antifa Ost](https://www.mdr.de/nachrichten/sachsen/dresden/dresden-radebeul/prozess-antifa-ost-kronzeuge-linksextremismus-100.html)
    *   **LS:** [Antifa-Prozess in Leichter Sprache](https://www.mdr.de/nachrichten-leicht/leichte-sprache-sachsen-antifa-prozess-100.html)
*   **Zusatzinfo:** Sitemaps unter `/index-sitemap.xml` erleichtern das Finden der Seiten massiv.

### taz (taz.de)
*   **Status:** `Gut geeignet`
*   **Strategie (Toborek / Eigene Analyse):** Die taz platziert den Hinweis auf den Originalartikel oft in einem kursiv gesetzten Absatz (`<em>`) am Ende des Textes. Der Crawler muss im LS-Artikel nach Links innerhalb von `<em>`-Tags suchen, deren Text "aus diesem „schweren“ Text" o.ä. lautet.
*   **Beispielpaar:**
    *   **AS:** [Barrierefreie Kommunikation im Netz](https://taz.de/Barrierefreie-Kommunikation-im-Netz/!5619787/)
    *   **LS:** [Leichte Sprache in der taz](https://taz.de/Leichte-Sprache/!5634433/)

### Stadt Köln
*   **Status:** `Sehr gut geeignet` (Umfangreiche Sammlung an Dienstleistungen)
*   **Strategie (Toborek):** Suche nach einem Link, dessen Text exakt `"Diese Seite in Alltags-Sprache lesen"` lautet (Groß-/Kleinschreibung ignorieren).
*   **Beispielpaar:**
    *   **AS:** [Pflege-Eltern werden](https://www.stadt-koeln.de/service/produkt/pflege-eltern-werden)
    *   **LS:** [Pflege-Eltern werden (LS)](https://www.stadt-koeln.de/leben-in-koeln/soziales/pflege-eltern-werden)

### Apotheken Umschau
*   **Status:** `Gut geeignet`
*   **Strategie (Toborek):** Suche in jedem LS-Artikel nach einem Link, der das Wort `"hier"` im `title`-Attribut trägt. Dieser Link führt zur Standard-Version.
*   **Beispielpaar:**
    *   **AS:** [Verhütung: Die Pille](https://www.apotheken-umschau.de/gesund-bleiben/sex/verhuetung-die-pille-707733.html)
    *   **LS:** [Pille](https://www.apotheken-umschau.de/einfache-sprache/verhuetung/pille-805349.html)

### Der Behindertenbeauftragte
*   **Status:** `Gut geeignet`
*   **Strategie (Toborek):** Suche gezielt nach einem Link mit der HTML-Klasse `.c-language-switch__l--as` und dem Text "Alltagssprache". Validierung über Regex: Der Link-Titel sollte den Satz "Lesen Sie den Artikel ... in Alltagssprache" enthalten.
*   **Beispielpaar:**
    *   **AS:** [Aufgabe des Beauftragten](https://www.behindertenbeauftragter.de/DE/DerBeauftragte/DieAufgabe/Aufgabe_node.html)
    *   **LS:** [Aufgabe des Beauftragten (LS)](https://www.behindertenbeauftragter.de/DE/LS/DerBeauftragte/DerBeauftragte_node.html)
*   **Alternative URL-Strategie:** Die URL-Pfade sind oft parallel aufgebaut (z.B. `/DE/DerBeauftragte/` vs. `/DE/LS/DerBeauftragte/`).

### Sozialpolitik.com
*   **Status:** `Gut geeignet`
*   **Strategie (Toborek):** Suche nach einem Link mit der Klasse `underline easy`, der explizit den Text `"Standardsprache"` enthält und auf die deutsche Version (`hreflang="de-DE"`) verweist.
*   **Beispielpaar:**
    *   **AS:** [Recht auf soziale Entschädigung](https://www.sozialpolitik.com/es/recht-auf-soziale-entschaedigung)
    *   **LS:** [Opferentschaedigung (LS)](https://www.sozialpolitik.com/opferentschaedigung)
*   **Alternative URL-Strategie:** Der URL der Leichten Sprache wird oft einfach das Präfix `ls-` vorangestellt.

### Lebenshilfe Main-Taunus
*   **Status:** `Gut geeignet` (Achtung: oft Einfache Sprache, nicht zwingend zertifizierte Leichte Sprache)
*   **Strategie (Toborek):** Suche im Bereich `mod_menue_top` nach einem Link mit dem Titel `"Auf Alltags-Sprache umstellen"`.
*   **Beispielpaar:**
    *   **AS:** [Bücherei](https://www.lebenshilfe-main-taunus.de/buecherei-74.html)
    *   **LS:** [Bücherei (LS)](https://www.lebenshilfe-main-taunus.de/ls/buecherei-74.html)

### Hamburg.de
*   **Status:** `Evaluierung ausstehend`
*   **Strategie (Excel-Liste):** Direkte Übersetzungsmöglichkeit wird über einen Button ("Leichte Sprache") prominent angezeigt. Crawler muss die Button-Ziele auslesen.
*   **Beispielpaar:**
    *   **AS:** [Hamburg barrierefrei](https://www.hamburg.de/hamburg-barrierefrei/)
    *   **LS:** [Hamburg barrierefrei (LS)](https://www.hamburg.de/hamburg-barrierefrei/leichte-sprache/)

---

## 2. Strukturelle & Inhalts-basierte Verknüpfung

Quellen, bei denen die Sprachversionen nicht durch direkte externe Links, sondern durch Struktur, CSS oder Metadaten miteinander verknüpft sind.

### Brand Eins
*   **Status:** `Besonderheit`
*   **Strategie (Toborek):** Beide Sprachversionen stehen auf *derselben* URL. Die Unterscheidung erfolgt rein über CSS: Absätze, die in roter Farbe (`#ff0000`) formatiert sind, werden als Leichte Sprache extrahiert, alle anderen als Standardsprache.
*   **Beispielpaar:**
    *   **AS:** [URL_BRAND_EINS] (Inhalt im Standard-Stil)
    *   **LS:** [URL_BRAND_EINS] (Inhalt in Rot)

### Saarländischer Rundfunk (sr.de)
*   **Status:** `Bedingt geeignet` (Oft Video als AS, Text als LS)
*   **Strategie (Eigene Analyse):** Wenn es sich um Textartikel handelt, kann die Zusammengehörigkeit über identische Open Graph Images (`<meta property="og:image" content="..." />`) in den Metadaten beider Seiten verifiziert werden.
*   **Beispielpaar:**
    *   **AS:** [Mitarbeiter verprügelt](https://www.sr.de/sr/home/nachrichten/panorama/kommunaler_ordnungsdienst_neunkirchen_mitarbeiter_verpruegelt_100.html)
    *   **LS:** [Mitarbeiter verprügelt (LS)](https://www.sr.de/sr/home/nachrichten/nachrichten_einfach/ne_mitarbeiter_der_stadt_neunkirchen_verpruegelt_100.html#)

---

## 3. Semantisches Alignment (Fallback-Strategie)

Quellen, denen eine konsistente Verlinkung oder URL-Struktur fehlt. Hier müssen NLP-Methoden zum Matchen angewandt werden.

### Tagesschau
*   **Status:** `Schwer zu scrapen`
*   **Problematik:** Fehlende Verlinkung und inkonsistente URL-Strukturen. Hoher inhaltlicher Reduktionsgrad in der LS.
*   **Strategie:**
    1. Unabhängiges Crawlen aller LS- und AS-Artikel eines Zeitraums.
    2. Berechnung von Vektor-Embeddings (z.B. via LaBSE, LASER, multilingual-e5).
    3. Berechnung der Cosine Similarity. Dokumentenpaare mit sehr hohem Score werden als Match gewertet.
*   **Beispielpaar:**
    *   **AS:** [Bundestags-Aussteiger](https://www.tagesschau.de/inland/bundestagswahl/bundestags-aussteiger-100.html)
    *   **LS:** [Bundestagswahl in LS](https://www.tagesschau.de/inland/bundestagswahl/leichte-sprache/bundestagswahl-in-leichter-sprache-148.html)

---

## 4. PDFs und Isolierte Dokumente

Quellen, die keine standardisierten Web-Texte nutzen, sondern Downloads (PDFs) bereitstellen. Erfordert PDF-Parsing-Bibliotheken (z.B. PyPDF2, pdfplumber).

### Wahlprogramme (Europawahl / Bundestagswahl)
*   **Quellen:** CDU, SPD, B90/Die Grünen, Die Linke, FDP
*   **Strategie:** Download des Standard-Programms und des LS-Programms als PDF. Text-Extraktion und anschließendes Alignment via Document-Similarity oder manuelles/semi-automatisches Mapping.
*   **Beispielpaar (CDU 2019):**
    *   **AS:** [Europawahlprogramm](https://www.cdu.de/system/tdf/media/dokumente/europawahlprogramm.pdf?file=1)
    *   **LS:** [Europawahlprogramm LS](https://www.cdu.de/system/tdf/media/dokumente/km_europawahlprogramm_leichte_sprache_2019.pdf?file=1&type=field_collection_item&id=18801)

### Bundesministerien (BMAS, BMFSFJ, etc.)
*   **Status:** `Schwankend` (Oft nur allgemeine Erklärungen, Publikationen oft als PDF).
*   **Strategie:** Manuelles Herausfiltern relevanter Publikationen (z.B. "Zweiter Teilhabebericht").
*   **Beispielpaar (BMAS):**
    *   **AS:** [Persönliches Budget Broschüre](https://www.bmas.de/DE/Service/Medien/Publikationen/a722-persoenliches-budget-broschuere.html)
    *   **LS:** [Persönliches Budget LS](https://www.bmas.de/DE/Service/Medien/Publikationen/a722-persoenliches-budget-broschuere.html)

