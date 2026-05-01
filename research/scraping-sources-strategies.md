# Übersicht der Scraping-Quellen und Alignment-Strategien (Leichte Sprache <-> Alltagssprache)

Dieses Dokument dient als "Living Document" (fortlaufend aktualisiert) für die Entwicklung von Web-Scrapern zur Erstellung eines parallelen Textkorpus (Standarddeutsch / Alltagssprache vs. Leichte/Einfache Sprache). Es aggregiert die Erkenntnisse aus unserer Recherche, dem Toborek-Paper (2023) und der analysierten Excel-Quellenliste.

## 1. Explizite Link-Verknüpfung (Direkte Backlinks)

Diese Quellen sind am verlässlichsten, da die Version in Leichter Sprache (LS) direkt auf die Version in Alltagssprache (AS) verlinkt (oder umgekehrt). Hier kann das Alignment 1:1 durch das Verfolgen von Links automatisiert werden.

### ✅ MDR (Mitteldeutscher Rundfunk)
*   **Status:** `Sehr gut geeignet`
*   **Strategie (Übersicht & Alignment):**
    1.  **Discovery:** Start auf der Übersichtsseite [`mdr.de/nachrichten-leicht/nachrichten-in-leichter-sprache-114.html`](https://www.mdr.de/nachrichten-leicht/nachrichten-in-leichter-sprache-114.html).
    2.  **Link-Extraktion:** Suche nach `div.box.cssArticle` oder `div.box.cssInfoTeaser`. Die Links zu den LS-Artikeln befinden sich in `h4 a.headline`.
    3.  **Alignment (Toborek / Eigene Analyse):** Suche im LS-Artikel nach einem Teaser-Block für die "schwere" Version. Spezifisch nach einem Element mit der Klasse `conHeadline` und dem Text `"Hier können Sie diese Nachricht auch in schwerer Sprache lesen:"`. Der darin liegende Link führt zum AS-Artikel.
    4.  **Content-Extraktion & Token-Zählung:**
        *   **Selektoren:** Um Rauschen (Menüs, Navigation, Footer) zu vermeiden, extrahiert der Scraper gezielt Inhalte aus `div.paragraph` und `p.text`.
        *   **Token-Zählung:** Erfolgt auf Basis von Whitespace-Splitting des bereinigten Fließtextes. Dies stellt sicher, dass nur der redaktionelle Inhalt in die Statistik einfließt.
*   **Beispielpaar:**
    *   **AS:** [Prozess Antifa Ost](https://www.mdr.de/nachrichten/sachsen/dresden/dresden-radebeul/prozess-antifa-ost-kronzeuge-linksextremismus-100.html)
    *   **LS:** [Antifa-Prozess in Leichter Sprache](https://www.mdr.de/nachrichten-leicht/leichte-sprache-sachsen-antifa-prozess-100.html)
*   **Zusatzinfo:** Sitemaps unter `/index-sitemap.xml` erleichtern das Finden der Seiten massiv.

### ✅ taz (taz.de)
*   **Status:** `Gut geeignet`
*   **Strategie (Übersicht & Alignment):**
    1.  **Discovery:** Start auf der Übersichtsseite [`taz.de/Politik/Deutschland/Leichte-Sprache/!p5097/`](https://taz.de/Politik/Deutschland/Leichte-Sprache/!p5097/).
    2.  **Link-Extraktion:** Suche direkt nach Links mit der Klasse `a.teaser-link`, die `Leichte-Sprache` in der URL enthalten. (URLs müssen oft am Semikolon `;` gekürzt werden).
    3.  **Alignment (Toborek / Eigene Analyse):** Die taz platziert den Hinweis auf den Originalartikel oft am Ende des Textes. Der Crawler sucht nach Links (`<a>`), die den Text `"schweren Text"` (oder ähnlich) enthalten, ODER nach Links, die innerhalb von kursiven Absätzen (`<em>`) liegen.
    4.  **Content-Extraktion & Token-Zählung:**
        *   **Selektoren:** Die taz verwendet oft `p.bodytext`, `p.article` oder legt den Inhalt in `<article itemprop="articleBody">` ab. Diese werden extrahiert und gefiltert (z. B. Ausschluss von Navigation, Footer und Trennlinien wie `──────────────────`).
        *   **Token-Zählung:** Analog zum MDR erfolgt die Zählung nach Entfernen von HTML-Tags über Whitespace-Splitting.
    5.  **Besonderheit (n:m Mapping):** Es gibt LS-Artikel, die Informationen aus mehreren AS-Artikeln zusammenfassen (z. B. „kommen aus diesem, diesem und diesem ‚schweren‘ Text“). Hier extrahiert das Skript alle passenden URLs und summiert die Tokens der AS-Artikel.
*   **Beispielpaar:**
    *   **AS:** [Migrantisches Leben in Dresden](https://taz.de/Migrantisches-Leben-in-Dresden/!5613086/)
    *   **LS:** [Döner aus Dresden](https://taz.de/Leichte-Sprache/!5617312/)

### ✅ Stadt Köln
*   **Status:** `Sehr gut geeignet` (Umfangreiche Sammlung an Dienstleistungen)
*   **Strategie (Übersicht & Alignment):**
    1.  **Discovery:** Da die Original-Übersicht nicht mehr aktiv ist, Nutzung des Wayback-Archivs: [`web.archive.org/.../informationen-leichter-sprache`](https://web.archive.org/web/20220804230818/https://www.stadt-koeln.de/leben-in-koeln/soziales/informationen-leichter-sprache).
    2.  **Link-Extraktion:** Suche in `ul.textteaserliste` nach Links (`li a.linkintern`). Diese führen zu den LS-Artikeln.
    3.  **Alignment (Toborek):** Suche im LS-Artikel nach einem Link, dessen Text exakt `"Diese Seite in Alltags-Sprache lesen"` lautet (Groß-/Kleinschreibung ignorieren).
    4.  **Content-Extraktion & Token-Zählung:**
        *   **Selektoren:** Der Hauptinhalt wird aus dem `<main id="inhalt">`-Element extrahiert. Es werden `p`, `li`, `h2` und `h3` Tags berücksichtigt.
        *   **Bereinigung:** Meta-Texte (wie der Alignment-Link selbst) und Wayback-Banner werden ignoriert.
    5.  **Technische Besonderheit (Wayback Machine):** Da archive.org bei hoher Last Verbindungen abbricht (`Connection refused`), muss der Scraper eine **Retry-Logik mit Exponential Backoff** (z. B. 2s, 4s, 8s) und einen höheren Delay zwischen Anfragen (mind. 4s) nutzen.
*   **Beispielpaar:**
    *   **AS:** [Ausweis-Papiere verloren?](http://www.stadt-koeln.de/leben-in-koeln/soziales/ausweis-papiere-verloren)
    *   **LS:** [Verlust von Ausweispapieren](http://www.stadt-koeln.de/service/produkt/verlust-von-ausweispapieren-1)

### ✅ Apotheken Umschau
*   **Status:** `Gut geeignet`
*   **Strategie (Übersicht & Alignment):**
    1.  **Discovery (Hierarchisch):** 
        *   Start auf der Übersichtsseite [`apotheken-umschau.de/einfache-sprache/`](https://www.apotheken-umschau.de/einfache-sprache/).
        *   Extrahiere alle Rubriken-Links (z.B. `/einfache-sprache/krankheiten/`, `/einfache-sprache/medikamente-heilpflanzen/`), typischerweise aus Elementen wie `a.stretched-link` in den Themenkacheln.
        *   Navigiere in diese Unterseiten und extrahiere dort die Links zu den einzelnen Artikeln, die auf `.html` enden (z.B. `href="/einfache-sprache/krankheiten/adipositas-769803.html"`).
    2.  **Alignment (Flexible Suche):** Suche in jedem LS-Artikel nach einem Link, der auf einen Artikel *außerhalb* des `/einfache-sprache/`-Verzeichnisses verweist. Der Link muss das Wort `"hier"` ODER `"informationen"` im `title`-Attribut tragen (z.B. `"mehr Informationen über die Krankheit"`).
    3.  **Content-Extraktion & Token-Zählung:**
        *   **Selektoren:** Fließtext befindet sich in `<article>` oder `<div class="article-body">`. Extrahiert werden die Tags `p` (oft als `p.text`), `h2`, `h3` und `li`.
        *   **Bereinigung:** Navigations-Elemente, Inhaltsverzeichnisse (`.article-chapter`) und Meta-Hinweise (wie "Dieser Link führt aus unserem Angebot heraus") werden strikt herausgefiltert.
*   **Beispielpaar:**
    *   **AS:** [Verhütung: Die Pille](https://www.apotheken-umschau.de/gesund-bleiben/sex/verhuetung-die-pille-707733.html)
    *   **LS:** [Pille](https://www.apotheken-umschau.de/einfache-sprache/verhuetung/pille-805349.html)

### ✅ Der Behindertenbeauftragte
*   **Status:** `Sehr gut geeignet` (fast flächendeckendes Angebot)
*   **Strategie (Übersicht & Alignment):**
    1.  **Discovery (Suche/Archiv & Pagination):** Die effizienteste Methode ist die Nutzung der Expertensuche unter `behindertenbeauftragter.de/SiteGlobals/Forms/Suche/Expertensuche_Formular.html` mit dem expliziten Filter `documentLanguage_str=de_ls`.
        *   **Artikel-Extraktion:** Artikel befinden sich in der Liste `ul.searchresult > li.teaser`.
        *   **Pagination:** Der Scraper extrahiert auf jeder Ergebnisseite den Link für die nächste Seite über das Element `a.forward.button` und wiederholt dies, bis kein "vor"-Button mehr existiert.
    2.  **Alignment (Language Switcher):** Auf nahezu jedem Artikel gibt es einen direkten Sprachwechsler. Die Strategie sucht vom LS-Artikel ausgehend gezielt nach einem Link mit der HTML-Klasse `.c-language-switch__l--as` (für Alltagssprache). Als Fallback wird nach einem Link mit dem Text "Alltagssprache" gesucht.
    3.  **Content-Extraktion & Token-Zählung:**
        *   **Selektoren:** Der Fließtext wird primär aus dem Container `<div id="content">` extrahiert (Tags: `p`, `h1`, `h2`, `h3`, `li`).
        *   **Bereinigung:** Essentiell ist das Filtern von Navigationselementen (`.c-mobile-nav__link`) und insbesondere des Sprachwechslers selbst (`.c-language-switch__li`, `.c-language-switch`), um zu verhindern, dass die Meta-Texte ("Lesen Sie den Artikel in Alltagssprache") als Tokens gezählt werden.
*   **Beispielpaar:**
    *   **AS:** [Lebenslauf Jürgen Dusel](https://www.behindertenbeauftragter.de/DE/AS/der-beauftragte/lebenslauf/lebenslauf-node.html)
    *   **LS:** [Lebenslauf Jürgen Dusel (LS)](https://www.behindertenbeauftragter.de/DE/LS/der-beauftragte/lebenslauf/lebenslauf-node.html)

### ✅ Sozialpolitik.com
*   **Status:** `Gut geeignet`
*   **Strategie (Übersicht & Alignment):**
    1.  **Discovery:** Start auf der Sitemap/Seiten-Übersicht für Leichte Sprache: [`sozialpolitik.com/es/seiten-uebersicht`](https://www.sozialpolitik.com/es/seiten-uebersicht). Dort sind alle verfügbaren LS-Artikel aufgelistet. Extrahiere alle Links, die mit `/es/` beginnen. Generische oder administrative Seiten (z.B. Impressum, Datenschutz, Kontakt) werden herausgefiltert.
    2.  **Alignment (Toborek):** Suche im LS-Artikel nach einem Link mit der Klasse `underline easy`, der explizit den Text `"Standardsprache"` (oder `"Inhalte für Standardsprache"`) enthält und auf die deutsche Version (`hreflang="de-DE"`) verweist.
    3.  **Content-Extraktion & Token-Zählung:**
        *   **Selektoren:** Der redaktionelle Fließtext wird direkt aus dem `<main>`-Container extrahiert (Tags: `p`, `li`, `h1`, `h2`, `h3`).
        *   **Bereinigung:** Navigationselemente (`.header-navigation-point`) und der Sprachwechsler selbst (`.underline.easy`) werden gefiltert, um die Zählung von Metainformationen zu vermeiden.
*   **Alternative URL-Strategie:** Oft ist das Alignment direkt über die URL nicht möglich, da die Titel in LS abweichen (z.B. `/arbeitswelt-von-morgen` in AS wird zu `/es/die-arbeits-welt` in LS). Daher ist die Extraktion über den Sprachwechsler essenziell.
*   **Beispielpaar:**

### ✅ Lebenshilfe Main-Taunus
*   **Status:** `Gut geeignet` (Achtung: oft Einfache Sprache, nicht zwingend zertifizierte Leichte Sprache)
*   **Strategie (Übersicht & Alignment):**
    1.  **Discovery:** Da die Hauptseite ins Archiv gewandert ist, nutze die "Inhalt"-Seite (Sitemap) in der Wayback-Machine: [`web.archive.org/.../lebenshilfe-main-taunus.de/ls/inhalt/`](https://web.archive.org/web/20200926190423/https://www.lebenshilfe-main-taunus.de/ls/inhalt/). Diese Seite listet alle verfügbaren `/ls/`-Links auf. Administrative Seiten (z.B. Impressum, Kontakt) sollten ausgeschlossen werden.
    2.  **Alignment (Toborek):** Suche im HTML nach einem Link mit dem Attribut `title="Auf Alltags-Sprache umstellen"`. Dieser führt zur Standard-Version.
    3.  **Content-Extraktion & Token-Zählung:**
        *   **Selektoren:** Der Hauptinhalt der Artikel befindet sich innerhalb von `<div class="inhalt">`. Es werden `p`, `li`, `h1`, `h2` und `h3` extrahiert.
        *   **Bereinigung:** Navigationsleisten (`nav`) und Sidebars (`#sidebar`) werden ignoriert. Typografische Artefakte der alten Website, wie gehäufte geschützte Leerzeichen (`\xa0`), werden durch normale Leerzeichen ersetzt.
    4.  **Technische Besonderheit (Wayback Machine):** Da die Seite komplett über archive.org gescrapt wird und diese restriktive Rate-Limits hat, ist eine starke Fehlerbehandlung zwingend erforderlich (z.B. 5 Retries, Exponentieller Backoff mit Faktor 2, und mindestens 5 Sekunden Pause zwischen den Artikelaufrufen), um `Connection refused` Fehler zu minimieren.
*   **Beispielpaar:**
    *   **AS:** [Bücherei](https://www.lebenshilfe-main-taunus.de/buecherei-74.html)
    *   **LS:** [Bücherei (LS)](https://www.lebenshilfe-main-taunus.de/ls/buecherei-74.html)

### ✅ Hamburg.de
*   **Status:** `Sehr gut geeignet`
*   **Strategie (Übersicht & Alignment):**
    1.  **Discovery (Hierarchisch):** 
        *   Start auf der Übersichtsseite [`hamburg.de/barrierefrei/leichte-sprache`](https://www.hamburg.de/barrierefrei/leichte-sprache).
        *   Extrahiere die Kategorie-Links (z. B. `/politik`, `/verkehr`, `/wohnen`), typischerweise Links mit genau 3 Slashes im Pfad.
        *   Navigiere in diese Unterseiten und extrahiere dort die Artikel-Links (Selektor: `a.km1-teaser__heading-link`).
    2.  **Alignment (Language Switcher):** Auf jedem LS-Artikel gibt es eine Sprachauswahl-Leiste (`.km1-language-bar__language`). Suche dort nach einem Link, der ein SVG-Icon mit der Klasse `.km1-icon--original-language` enthält. Dieser Link führt zur Version in Standardsprache.
    3.  **Content-Extraktion & Token-Zählung:**
        *   **Selektoren:** Da Artikel oft aus mehreren Textblöcken bestehen, müssen *alle* Elemente mit der Klasse `.km1-richtext` extrahiert werden.
        *   **Fehler-Erkennung:** Aufgrund einer kürzlichen Umstrukturierung des Portals führen einige Alignment-Links zu "Seite nicht gefunden (Fehler 404)". Der Scraper muss diese Texte erkennen und das Alignment in solchen Fällen verwerfen.
*   **Beispielpaar:**
    *   **AS:** [Der Senat](https://www.hamburg.de/politik-und-verwaltung/senat/senat-236762)
    *   **LS:** [Was ist der Senat?](https://www.hamburg.de/barrierefrei/leichte-sprache/politik/ls-der-senat-576194)

### ✅ Stuttgart.de
*   **Status:** `Sehr gut geeignet`
*   **Strategie (Übersicht & Alignment):**
    1.  **Discovery:** Start auf der Übersichtsseite [`stuttgart.de/leichte-sprache-index`](https://www.stuttgart.de/leichte-sprache-index). Extraktion aller Links, deren Text auf `(Leichte Sprache)` endet.
    2.  **Alignment (Duale Strategie):**
        *   **URL-Logik (Primär):** Die AS-Version ist in der Regel identisch mit der LS-URL, jedoch ohne den Query-Parameter `?sp:out=easy` (bzw. `?sp%3Aout=easy`).
        *   **Link-Logik (Sekundär/Verifikation):** In jedem LS-Artikel findet sich ein Link mit dem Attribut `aria-label="Artikel in Alltags-Sprache"` (Klasse `.SP-Link`), der direkt auf die AS-Version verweist. Umgekehrt findet sich auf AS-Seiten ein entsprechender Link zur Leichten Sprache.
    3.  **Content-Extraktion & Token-Zählung:**
        *   **Selektoren:** Der Hauptinhalt wird aus dem `<main>`-Tag extrahiert.
        *   **Bereinigung:** Entfernen von Funktionselementen wie `.SP-Intro__tools` (Sprachwechsler-Box) und `.SP-Linklist` (verwandte Links), um die Token-Zählung nicht zu verfälschen.
*   **Beispielpaar:**
    *   **AS:** [Personalausweis beantragen](https://www.stuttgart.de/organigramm/leistungen/personalausweis-beantragen-erstmalig-oder-nach-ablauf)
    *   **LS:** [Personalausweis beantragen (LS)](https://www.stuttgart.de/organigramm/leistungen/personalausweis-beantragen-erstmalig-oder-nach-ablauf?sp%3Aout=easy)

---

## 2. Strukturelle & Inhalts-basierte Verknüpfung

Quellen, bei denen die Sprachversionen nicht durch direkte externe Links, sondern durch Struktur, CSS oder Metadaten miteinander verknüpft sind.

### ✅ Brand Eins
*   **Status:** `Sehr gut geeignet` (Paralleltexte auf einer Seite)
*   **Strategie (Übersicht & Extraktion):**
    1.  **Discovery:** Da die Live-Seite von Brand Eins strenge Cloudflare-Blockaden für automatisierte Abrufe hat, muss die Übersichtsseite über die Wayback Machine geladen werden (z. B. ein Schnappschuss aus 2024 unter [`web.archive.org/.../brandeins.de/themen/rubriken/leichte-sprache`](https://web.archive.org/web/20240401000000/https://www.brandeins.de/themen/rubriken/leichte-sprache)).
    2.  **Link-Extraktion:** Suche nach allen Links (`<a>`), die auf `/magazine/` verweisen und keine reinen Systemlinks (wie `/products/` oder `/sign_in`) sind.
    3.  **Content-Extraktion & Alignment (Dual-Heuristik):** Da beide Sprachversionen auf derselben Seite im Element `section.textblock` stehen, müssen sie getrennt extrahiert werden:
        *   **Farben-Heuristik:** Suche im `style`-Attribut der `<p>`-Tags nach den Rottönen der Leichten Sprache (`#fa4600`, `#ff4948`, `#ff0000`, `color: red`).
        *   **Struktur-Heuristik (Fallback):** Ist keine Farbe im HTML-Schnappschuss hinterlegt, greift die Struktur-Regel: In einem `textblock` ist der erste `<p>`-Tag meist der schwerere Originaltext (AS), alle folgenden `<p>`-Tags sind Leichte Sprache. AS-Absätze lassen sich zudem oft durch Auslassungszeichen `(…)` oder eine überdurchschnittliche Länge (> 300 Zeichen) verifizieren.
*   **Beispielpaar:**
    *   **AS & LS:** [Die Regierung zieht nicht mit um](https://www.brandeins.de/magazine/brand-eins-wirtschaftsmagazin/2024/kommunikation-in-zeiten-von-fake-news/die-regierung-zieht-nicht-mit-um) (LS ist der rot gesetzte Teil).

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

