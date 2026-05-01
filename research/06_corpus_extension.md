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
