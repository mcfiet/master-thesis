# Korpus-Erweiterung: Identifikation zusätzlicher Quellen

Dieses Dokument dokumentiert die Recherche und Evaluation neuer Textquellen für den Korpus, die über die in der Standard-Literatur (z.B. Klaper et al., 2013; Battisti, 2020) bereits bekannten Quellen hinausgehen.

## Zielsetzung der Woche
Das Ziel dieser Woche ist es, den Korpus quantitativ und qualitativ zu verbreitern, indem systematisch nach Webseiten gesucht wird, die:
1.  **Authentische Leichte Sprache** (nach BITV 2.0 oder Netzwerk Leichte Sprache) anbieten.
2.  Ein **direktes Alignment** zu alltagssprachlichen Texten ermöglichen.
3.  Bisher in der Forschung **wenig oder gar nicht** berücksichtigt wurden (um Overfitting auf bekannte Datensätze zu vermeiden).

---

## Recherche-Strategien

1. **Prüfung öffentlicher Stellen (Städte und Bundesländer):** Als erste Strategie gehe ich die Websites der öffentlichen Stellen von Städten und Bundesländern durch. Da diese gesetzlich verpflichtet sind, wesentliche Informationen in Leichter Sprache bereitzustellen, bieten sie ein hohes Potenzial für stabiles URL-Alignment. Ein erfolgreiches Beispiel hierfür ist `hamburg.de`, weshalb dieses Vorgehen nun systematisch auf andere Metropolen und Landesportale ausgeweitet wird.

---

## Potenzielle neue Quellen (Longlist)

| Bundesland | URL (Land) | Landeshauptstadt | URL (Stadt) |
| :--- | :--- | :--- | :--- |
| Baden-Württemberg | [baden-wuerttemberg.de](https://www.baden-wuerttemberg.de/) | Stuttgart | [stuttgart.de](https://www.stuttgart.de/) |
| Bayern | [bayern.de](https://www.bayern.de/) | München | [muenchen.de](https://www.muenchen.de/) |
| Berlin | [berlin.de](https://www.berlin.de/) | - | - |
| Brandenburg | [brandenburg.de](https://www.brandenburg.de/) | Potsdam | [potsdam.de](https://www.potsdam.de/) |
| Bremen | [bremen.de](https://www.bremen.de/) | - | - |
| Hamburg | [hamburg.de](https://www.hamburg.de/) | - | - |
| Hessen | [hessen.de](https://www.hessen.de/) | Wiesbaden | [wiesbaden.de](https://www.wiesbaden.de/) |
| Mecklenburg-Vorpommern | [mecklenburg-vorpommern.de](https://www.mecklenburg-vorpommern.de/) | Schwerin | [schwerin.de](https://www.schwerin.de/) |
| Niedersachsen | [niedersachsen.de](https://www.niedersachsen.de/) | Hannover | [hannover.de](https://www.hannover.de/) |
| Nordrhein-Westfalen | [land.nrw](https://www.land.nrw/) | Düsseldorf | [duesseldorf.de](https://www.duesseldorf.de/) |
| Rheinland-Pfalz | [rlp.de](https://www.rlp.de/) | Mainz | [mainz.de](https://www.mainz.de/) |
| Saarland | [saarland.de](https://www.saarland.de/) | Saarbrücken | [saarbruecken.de](https://www.saarbruecken.de/) |
| Sachsen | [sachsen.de](https://www.sachsen.de/) | Dresden | [dresden.de](https://www.dresden.de/) |
| Sachsen-Anhalt | [sachsen-anhalt.de](https://www.sachsen-anhalt.de/) | Magdeburg | [magdeburg.de](https://www.magdeburg.de/) |
| Schleswig-Holstein | [schleswig-holstein.de](https://www.schleswig-holstein.de/) | Kiel | [kiel.de](https://www.kiel.de/) |
| Thüringen | [thueringen.de](https://www.thueringen.de/) | Erfurt | [erfurt.de](https://www.erfurt.de/) |

---

## Detaillierte Ergebnisse der Korpus-Erweiterung

In dieser Sektion werden die Ergebnisse für die neu erschlossenen Quellen detailliert dokumentiert, analog zur Methodik in Dokument `05`.

### 1. Stuttgart.de

#### Status & Statistiken
*   **Skripte:** `scraper/stuttgart_scraper.py` (Alignment) & `corpus_scrapers/stuttgart_scraper.py` (Extraktion)
*   **Ergebnis:** 42 valide Artikel-Paare.
*   **Token-Verteilung:**
    *   Gesamt LS: 23.843 Tokens (Ø 567 pro Artikel)
    *   Gesamt AS: 46.122 Tokens (Ø 1.098 pro Artikel)
    *   Verhältnis (AS/LS): ~1,93 (Sehr gute Datenlage für Vereinfachungs-Analysen)

#### Strategie (Übersicht & Alignment)
1.  **Discovery:** Start auf der zentralen Übersichtsseite [`stuttgart.de/leichte-sprache-index`](https://www.stuttgart.de/leichte-sprache-index). Extraktion aller Links, die das Label `(Leichte Sprache)` im Text führen.
2.  **Alignment (Duale Strategie):**
    *   **URL-Logik:** Die AS-Version wird primär durch das Entfernen des Query-Parameters `?sp:out=easy` (bzw. `?sp%3Aout=easy`) aus der LS-URL abgeleitet.
    *   **Explizite Links:** Auf den Seiten existieren zudem explizite Sprachwechsler-Elemente (`.SP-Link` mit `aria-label="Artikel in Alltags-Sprache"`), die zur Verifikation dienen können.
3.  **Content-Extraktion:** Gezielte Extraktion aus dem `<main>`-Tag, wobei die logische Struktur (`p`, `li`, `h1-3`) beibehalten wird.

#### Herausforderungen & Verbesserungen
*   **Initialer Status:** Erhebliche Mengen an Boilerplate in den Rohdaten (Sharing-Tools, Kontaktboxen, Stand-Zeitstempel). Repetitive Phrasen wie "Seite teilen" oder "Das könnte Sie auch interessieren" verfälschten die Wortschatz-Statistik.
*   **Surgical Cleaning:**
    *   **Container-Decomposition:** Gezieltes Löschen von CSS-Klassen wie `.SP-Intro__tools`, `.SP-ContentFooter`, `.SP-Share` und `.SP-JumboButton__container`.
    *   **Text-Filter:** Implementierung einer Blacklist für repetitive Sätze ("Übersetzt und geprüft vom...", "Öffnet in einem neuen Tab") und Meta-Daten-Fragmente (PDF-Größenangaben).
    *   **Token-Präzision:** Durch die Bereinigung sank das Rauschen im LS-Teil um ca. 10%, was zu einem authentischeren linguistischen Profil führt.

#### Beispielpaar
*   **AS:** [Reisepass beantragen](https://www.stuttgart.de/organigramm/leistungen/reisepass-beantragen-erstmalig-oder-nach-ablauf)
*   **LS:** [Reisepass beantragen (LS)](https://www.stuttgart.de/organigramm/leistungen/reisepass-beantragen-erstmalig-oder-nach-ablauf?sp%3Aout=easy)

### 2. Wiesbaden.de

#### Status & Statistiken
*   **Skripte:** `scraper/wiesbaden_scraper.py` (Alignment) & `corpus_scrapers/wiesbaden_scraper.py` (Extraktion)
*   **Ergebnis:** 41 Artikel-Paare nach Bereinigung (ursprünglich 44).
*   **Token-Verteilung:**
    *   Gesamt LS: 7.010 Tokens (Ø 171 pro Artikel)
    *   Gesamt AS: 10.100 Tokens (Ø 246 pro Artikel)
    *   Verhältnis (LS/AS): ~0,69

#### Strategie (Übersicht & Alignment)
1.  **Discovery:** Ähnlich wie bei Stuttgart (beide Städte nutzen offenbar das gleiche CMS/System), erfolgt das Alignment durch Anhängen oder Entfernen des Query-Parameters `?sp:easylanguage=1` (bzw. `?sp%3Aeasylanguage=1`).
2.  **Content-Extraktion:** Extraktion zielt auf den Hauptcontainer (`article#SP-Content` oder `div.SP-Content__body`), unter Beibehaltung von Headern, Absätzen und Listen.

#### Herausforderungen & Verbesserungen beim Scraper
*   **UI-Rauschen:** Das initial generierte Corpus wies zahlreiche UI-Fragmente ("(Öffnet in einem neuen Tab)", Navigations- und Service-Links wie "Zum Fahrplan" oder "Routenplaner öffnen") auf. 
*   **"Stubs" und leere Seiten:** Einige LS-Seiten enthielten so gut wie keinen Text (z.B. nur 2 bis 10 Wörter), während die dazugehörigen AS-Seiten extrem ausführlich waren.
*   **Lösung:** Der Extraktor wurde deutlich verschärft:
    *   Es wurde ein **Quality Filter** eingeführt: Nur Paare mit mindestens 40 Token auf der LS-Seite werden gespeichert.
    *   Spezifische Blacklists für Navigationsphrasen wurden eingebaut und CSS-Klassen (`.SP-Link--simple-language`, `.SP-Navigation`) gezielt entfernt.
    *   Das Resultat ist ein formal sehr sauberes Corpus ohne störendes Boilerplate.

#### Strukturelle Probleme (Korpus-Qualität)
Trotz des erfolgreichen Scrapings und der tiefen Bereinigung hat die manuelle Inspektion **gravierende strukturelle Probleme auf Seiten des Wiesbaden-Portals** offengelegt, die das Corpus für das Training eines KI-Modells weitestgehend unbrauchbar machen:
*   **Mangelndes Alignment / Themenabweichung:** Die Redaktion pflegt unter der LS-URL oft Inhalte, die **nichts oder nur peripher** mit der Original-URL zu tun haben. Es handelt sich oft nicht um Text-zu-Text-Übersetzungen.
    *   *Beispiel 1:* Auf der LS-Seite geht es um den *ÖKOPROFIT-Klub* für Unternehmen, auf der AS-Seite um ein *Energieeffizienz-Netzwerk*.
    *   *Beispiel 2:* Die LS-Seite erklärt generisch den *Waldnaturschutz*, während die AS-Seite die historische Entwicklung (*Wachstum des Waldes seit dem 19. Jahrhundert*) behandelt.
    *   *Beispiel 3:* Ein LS-Text erklärt anschaulich den Begriff *Smart City*, der AS-Text ist hingegen lediglich eine *Pressemitteilung zu einer Poster-Kampagne*.
*   **CMS Mapping-Fehler:** In einigen Fällen führt die LS-URL zu völlig falschen Inhalten (z.B. Murnau-Filmtheater LS-Seite enthält den Text über die Caligari FilmBühne).

**Fazit:** Der Scraper extrahiert technisch perfekt, aber die Ausgangsdaten von `wiesbaden.de` sind semantisch nicht parallel. Ein Modell würde hier fälschlicherweise lernen, dass "Leichte Sprache" bedeutet, über inhaltlich völlig andere Fakten zu sprechen. Dieser Teilkorpus sollte bei der Modell-Trainingsphase mit großer Vorsicht genossen oder gänzlich ausgeschlossen werden.

### 3. Hannover.de

#### Status & Statistiken
*   **Skripte:** `scraper/hannover_scraper.py` (Alignment) & `corpus_scrapers/hannover_scraper.py` (Extraktion)
*   **Ergebnis:** 846 valide Artikel-Paare.
*   **Token-Verteilung:**
    *   Gesamt LS: 473.305 Tokens (Ø 559 pro Artikel)
    *   Gesamt AS: 400.422 Tokens (Ø 473 pro Artikel)
    *   Verhältnis (LS/AS): ~1,18 (LS-Texte sind oft länger, da mehr erklärender Kontext hinzugefügt wird)

#### Rechtlicher Hintergrund (Niedersachsen)
Die Bereitstellung von Inhalten in Leichter Sprache auf `hannover.de` ist stark durch das **Niedersächsische Behindertengleichstellungsgesetz (NBGG)** geprägt. Speziell **§ 9a NBGG** (in Verbindung mit der NBITVO) verpflichtet öffentliche Stellen in Niedersachsen zur Barrierefreiheit und dazu, wesentliche Informationen sowie Navigationshinweise in Leichter Sprache bereitzustellen. Im Gegensatz zu vielen anderen Behörden geht Hannover jedoch weit über das gesetzliche Minimum hinaus und übersetzt regelmäßig aktuelle Nachrichten und Serviceinformationen.

#### Strategie (Übersicht & Alignment)
1.  **Discovery (Rekursives Crawling):** Da die offiziellen XML-Sitemaps von hannover.de unvollständig waren und die LS-Artikel nicht erfassten, wurde eine rekursive Crawling-Strategie implementiert. Ausgehend von der Startseite `https://www.hannover.de/Leichte-Sprache` sammelt der Scraper systematisch alle internen Links innerhalb des `/Leichte-Sprache`-Pfades.
2.  **Alignment (Canonical-Link-Extraktion):** Ein struktureller Glücksfall auf der Website: Die Artikel im LS-Bereich enthalten verlässliche `<link rel="canonical">`-Tags, die direkt auf die alltagssprachliche Originalversion (`AS-URL`) verweisen. Der URL-Ausrichter besucht die LS-Seite, liest den Canonical-Tag aus und speichert das Paar, sofern der Link nicht in den LS-Bereich zurückführt.

#### Herausforderungen & Verbesserungen
*   **Mediopunkt-Nutzung:** Die LS-Texte auf hannover.de nutzen sehr konsequent den Mediopunkt (`∙`) zur Silbentrennung (z. B. `Bewohner∙park∙plätze`). Dies ist ein wichtiges Stilmittel der Leichten Sprache und wurde bei der Extraktion strikt beibehalten, da es wertvolle linguistische Merkmale für das Modelltraining liefert.
*   **Boilerplate und Rauschen:** Nach einem ersten Extraktionsdurchlauf zeigte eine Korpus-Analyse, dass die Daten extrem mit Boilerplate kontaminiert waren.
    *   *Problem:* Wiederkehrende Phrasen wie "Zur Seite in Alltagssprache", "Weitere Informationen in Leichter Sprache" oder "Auf dieser Seite erfahren Sie:" tauchten in fast jedem Textpaar auf. Dies hätte zu starkem Rauschen und Überanpassung (Overfitting) beim Modelltraining geführt.
    *   *Lösung:* Der Extraktor (`corpus_scrapers/hannover_scraper.py`) wurde mit einer umfassenden Blacklist-Filterung versehen. Durch String-Matching wurden diese Metatexte sowie UI-Elemente ("Drucken", "E-Mail") konsequent herausgefiltert. Nach dem Neuschreiben des Korpus stieg die semantische Qualität der Daten signifikant an.

---

## Zentrale Erkenntnis: Gesetzliche Verpflichtungen vs. Korpus-Realität

Im Zuge der Recherche zu den Portal-Strukturen von öffentlichen Stellen wurde deutlich, warum ein Großteil der Behörden-Websites zwar einen Bereich für "Leichte Sprache" besitzt, sich dort aber meist keine 1:1 übersetzten Artikel oder Blogposts für ein Parallelkorpus finden lassen.

### Rechtlicher Hintergrund (BITV 2.0 & BGG)
Die Verpflichtung zur Barrierefreiheit für öffentliche Stellen in Deutschland ist im **Behindertengleichstellungsgesetz (BGG)** und der **Barrierefreie-Informationstechnik-Verordnung (BITV 2.0)** verankert:

*   **[BITV 2.0 § 4 i.V.m. Anlage 2](https://www.gesetze-im-internet.de/bitv_2_0/__4.html):** Öffentliche Stellen sind verpflichtet, auf der Startseite ihrer Website Links zu folgenden Informationen in Leichter Sprache bereitzustellen:
    1.  **Wesentliche Inhalte:** Informationen über die Aufgaben und Angebote der Behörde.
    2.  **Navigation:** Erläuterungen zur Nutzung der Website.
    3.  **Erklärung zur Barrierefreiheit:** Informationen zum Stand der Barrierefreiheit, dem **Feedback-Mechanismus** (Möglichkeit zur Meldung von Mängeln) sowie Hinweise auf das **Schlichtungsverfahren**.
*   **Rechtswidrigkeit:** Eine Behörde handelt bereits dann rechtswidrig, wenn diese spezifischen Einstiegsinformationen nicht in Leichter Sprache vorhanden sind.
*   **Keine Vollübersetzungspflicht:** Wichtig für die Korpus-Erstellung ist die Erkenntnis, dass das Gesetz **keine Pflicht zur vollständigen Übersetzung** aller Inhalte (z.B. tägliche News, Fachartikel, Archiv-Inhalte) in Leichte Sprache vorsieht. Es ist lediglich gefordert, Informationen "vermehrt" in Leichter Sprache bereitzustellen (§ 11 BGG).

### Implikationen für die Forschung
Diese gesetzliche "Minimallösung" erklärt eine wesentliche Herausforderung bei der Datensatz-Erstellung:
1.  **Minimaler Erfüllungsgrad:** Ein Großteil der Behörden-Websites setzt exakt nur die gesetzlich geforderten Basistexte um (Einstiegsseite, Erklärung der Navigation und Barrierefreiheit).
2.  **Fehlendes 1:1 Alignment:** Es gibt auf den allermeisten dieser Seiten keine 1:1-Übersetzungen von tagesaktuellen Artikeln, Pressemitteilungen oder Blog-Posts. Selbst wenn ein "Leichte Sprache"-Button global im Website-Header verankert ist, führt dieser beim Klicken auf regulären Unterseiten oft nur zurück auf die immer gleiche, generische Übersichtsseite.
3.  **Quellenauswahl:** Für den Aufbau eines hochwertigen Parallelkorpus müssen Portale identifiziert werden, die proaktiv und deutlich über das gesetzliche Minimum hinausgehen (z.B. Nachrichtenseiten wie `mdr.de` oder Städte mit dedizierten LS-Redaktionen wie `hamburg.de` oder `stuttgart.de`). Ein rein automatisches Crawling von Standard-Behördenseiten führt meist ins Leere, da schlichtweg keine Text-Paare existieren.


## Nächste Schritte:

1. Datensatz deskriptiv beschreiben (genaue Datensatzanalyse)
2. mit z.B. Bert alignment checken
3. Zweitprüfer kontakieren: Peter John
