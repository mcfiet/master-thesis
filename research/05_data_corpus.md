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

## 3. Architektur der Korpus-Scraper (Allgemein)

Nach den ersten Analysen wurden alle Scraper im Verzeichnis `scripts/corpus_scrapers/` auf ein einheitliches **Downloader-Prinzip** umgestellt:

1.  **Input:** Die Skripte lesen die `*_aligned_urls.json` aus `results/aligned_urls/`.
2.  **Kein Crawling:** Es findet keine neue Suche nach URLs statt; das schont die Serverressourcen und erhöht die Geschwindigkeit.
3.  **Output:** Die bereinigten Daten werden nach `results/corpus/` geschrieben.
4.  **Einheitliche Reinigung:** Alle Skripte nutzen nun eine verbesserte Basis-Reinigung (Entfernung von Script-, Style- und Bild-Tags).

---

## Status der weiteren Quellen

Für die folgenden Quellen wurden die Scraper bereits auf das Downloader-Prinzip umgestellt. Eine detaillierte qualitative Analyse der extrahierten Texte steht hier noch aus und erfolgt analog zu den obigen Beispielen:

*   **Brand eins:** Extraktion beider Sprachstufen von einer einzigen URL (Heuristik basierend auf Textblöcken und Farbindikatoren).
*   **Stadt Köln:** Fokus auf das Haupt-Inhaltsverzeichnis der archivierten Wayback-Snapshots.
*   **MDR:** Extraktion der News-Beiträge inklusive Archiv-Suche.
*   **Taz:** Unterstützung von 1-zu-n Mappings (ein LS-Artikel referenziert oft mehrere AS-Artikel).
*   **Lebenshilfe Main-Taunus:** Extraktion aus dem lokalen Inhaltsverzeichnis `/ls/inhalt/`.
*   **Sozialpolitik.com:** Extraktion basierend auf der Sitemap-Übersicht.
