# Dataset Research: Quellenanalyse für Parallelkorpora (LS/AS)

Diese Untersuchung analysiert verschiedene Nachrichtenportale und Institutionen hinsichtlich ihrer Eignung zur Extraktion von Paralleltexten in Leichter Sprache (LS) und Alltagssprache (AS).

## 1. MDR (mdr.de)

Der MDR bietet eine systematische Struktur, die sich gut für automatisiertes Scraping eignet.

### Analyse & Struktur
- **Sitemaps:** Der MDR stellt für jede Unterseite Sitemaps über `/index-sitemap.xml` bereit, was das Crawlen erleichtert.
- **Verknüpfung:** Artikel in Alltagssprache und Leichter Sprache sind gegenseitig verlinkt. Auf beiden Versionen findet sich ein Button, der direkt zur entsprechenden Fassung des anderen Sprachniveaus führt.

### Technische Umsetzung (Linking Mechanism)
Die Verlinkung zur "schweren Sprache" ist in einem spezifischen Container eingebettet:

![[Pasted image 20260410205335.png]]

```html
<div class="con ...">
  <h3 class="conHeadline">Hier können Sie diese Nachricht auch in schwerer Sprache lesen:</h3>
  <div class="modCon">
     ...
     <a href="/ziel-url.html" class="linkAll"></a>
  </div>
</div>
```

### Beispiele
- **Alltagssprache (AS):** [Prozess Antifa Ost](https://www.mdr.de/nachrichten/sachsen/dresden/dresden-radebeul/prozess-antifa-ost-kronzeuge-linksextremismus-100.html)
- **Leichte Sprache (LS):** [Antifa-Prozess in Leichter Sprache](https://www.mdr.de/nachrichten-leicht/leichte-sprache-sachsen-antifa-prozess-100.html)

---

## 2. taz (taz.de)

Die taz bietet ebenfalls Paralleltexte, jedoch scheint die Verknüpfung weniger systematisch oder prominent platziert zu sein als beim MDR.

### Analyse & Struktur
- **Verknüpfung:** Bei einigen Artikeln in Leichter Sprache befindet sich am Ende des Textes ein manueller Link zur Standard-Version.

### Technische Umsetzung
Die Verlinkung erfolgt meist innerhalb eines `<em>`-Tags im Fließtext:

![[Pasted image 20260410205306.png]]

```html
<p class="bodytext ...">
  <em>
    <a href="/ziel-url/" class="link in-text-link">aus diesem „schweren“ Text</a>
  </em>
</p>
```

### Beispiele
- **Leichte Sprache (LS):** [Barrierefreie Kommunikation](https://taz.de/Leichte-Sprache/!5634433/)
- **Alltagssprache (AS):** [Netz-Kommunikation](https://taz.de/Barrierefreie-Kommunikation-im-Netz/!5619787/)

---

## 3. Saarländischer Rundfunk (SR / sr.de)

Beim SR variiert die Eignung der Artikel stark je nach Format.

### Analyse & Eignung
- **Eingeschränkt geeignet:** Oft ist der Artikel in Alltagssprache ein Video-Beitrag, während die Leichte Sprache als Text vorliegt. Dies erschwert das Alignment auf Satzebene.
- **Gut geeignet:** Bei reinen Textartikeln lässt sich die Zusammengehörigkeit oft über das `og:image` (Open Graph Image) verifizieren, da dieses in beiden Versionen identisch ist.

### Beispiele
- **AL** [MDR Video Beitrag](https://www.sr-mediathek.de/index.php?seite=7&id=166912&startvid=3)
- **LS:** [Beitrag in Text]([https://www.sr.de/sr/home/nachrichten/nachrichten_einfach/ne_mitarbeiter_der_stadt_neunkirchen_verprügelt_100.html#](https://www.mdr.de/nachrichten-leicht/leichte-sprache-sachsen-antifa-prozess-100.html))
- Nicht geeignet

---

## 4. Weitere Institutionen & Portale

### Lebenshilfe Main-Taunus
- **Mechanismus:** Bietet auf jeder Unterseite einen prominenten Button für "EINFACHE SPRACHE".
- **URL:** [lebenshilfe-main-taunus.de](https://www.lebenshilfe-main-taunus.de/)
- Einfache Sprache ≠ Leichte Sprache

### GWW-Netz
- **Mechanismus:** Besitzt einen Sprachumschalter direkt auf der Startseite.
- **URL:** [gww-netz.de](https://www.gww-netz.de/)
- **Beispiel AS:** [Gesundheit & Pflege Forum](https://www.gww-netz.de/de/aktuelles/magazin/gesundheit-pflege-digitalisierung-interdisziplinaeres-forum-bei-der-gww.html)
- **Beispiel LS:** [Probleme in der Pflege](https://www.gww-netz.de/de-LS/aktuelles/magazin/zenit-bespricht-probleme-in-der-pflege.html)

---

## 5. Wörterbücher & Ressourcen

Für terminologische Analysen oder als Ergänzung zum Korpus können Fachwörterbücher genutzt werden:
- **Hurraki:** [Wiki für Leichte Sprache (A-Z)](https://hurraki.de/wiki/Hurraki:Artikel_von_A_bis_Z#A)
- **Nachrichtenleicht:** [Wörterbuch-Sektion](https://www.nachrichtenleicht.de/woerterbuch)

---

## 6. Wissenschaftliche Referenzen

Eine aktuelle Zusammenfassung bestehender Korpora in Leichter Sprache findet sich in folgendem Paper:
- [ACM Digital Library - Research on Easy-to-Read German](https://dl.acm.org/doi/fullHtml/10.1145/3594806.3596530#BibPLXBIB0008)

---

## 7. A New Aligned Simple German Corpus (Toborek et al., 2023)

Dieses Projekt wurde auf der ACL 2023 vorgestellt und bietet eine fundierte Grundlage für die Erstellung eines Parallelkorpus für Deutsch und Leichte/Einfache Sprache. Das zugehörige Repository enthält sowohl die Scraper als auch Algorithmen zum Alignment auf Satzebene.

### Hintergrund & Zielsetzung
Ziel der Arbeit von Toborek et al. ist es, die Datenlage für maschinelle Übersetzung (NMT) ins Deutsche "Leichte Sprache" zu verbessern. Da bestehende Korpora oft klein oder nicht frei verfügbar sind, automatisiert dieses Projekt das Scraping von Webseiten, die parallele Versionen anbieten, und nutzt verschiedene Matching-Algorithmen, um korrespondierende Sätze zu identifizieren.

### Alignment-Strategien der Quellen
In diesem Repository nutzen die Crawler je nach Nachrichtenquelle unterschiedliche Strategien, um Artikel in Leichter Sprache (LS) und Standardsprache (Standard) einander zuzuordnen:

1.  **Apotheken Umschau**
    *   **Strategie:** Suche nach einem spezifischen Link im LS-Artikel.
    *   **Details:** Der Crawler sucht in jedem LS-Artikel nach einem Link, der das Wort "hier" im `title`-Attribut trägt. Dieser Link führt direkt zur Standard-Version des Artikels.
2.  **Behindertenbeauftragter**
    *   **Strategie:** Nutzung des Sprachumschalters auf der Webseite.
    *   **Details:** Es wird gezielt nach einem Link mit der Klasse `c-language-switch__l--as` und dem Text "Alltagssprache" gesucht. Ein regulärer Ausdruck validiert zudem, dass der Link-Titel den Satz "Lesen Sie den Artikel ... in Alltagssprache" enthält.
3.  **Brand Eins**
    *   **Strategie:** Extraktion beider Versionen aus einem einzigen Dokument.
    *   **Details:** Eine Besonderheit dieser Quelle. Beide Sprachversionen stehen auf derselben URL. Der Crawler unterscheidet sie anhand von CSS-Styles: Absätze, die in roter Farbe (`#ff0000`) formatiert sind, werden als Leichte Sprache extrahiert, alle anderen als Standardsprache.
4.  **Lebenshilfe Main-Taunus**
    *   **Strategie:** Suche nach einem Umstell-Link im Kopfmenü.
    *   **Details:** In jedem LS-Artikel wird im Bereich `mod_menue_top` nach einem Link mit dem Titel "Auf Alltags-Sprache umstellen" gesucht.
5.  **MDR (Mitteldeutscher Rundfunk)**
    *   **Strategie:** Suche nach einem Teaser-Block für die "schwere" Version.
    *   **Details:** Der Crawler sucht nach einem Element, das den Text "auch in schwerer Sprache" enthält. Der darin enthaltene Link führt zum Standardartikel.
6.  **Sozialpolitik.com**
    *   **Strategie:** Identifikation über Link-Klasse und Text.
    *   **Details:** Der Crawler sucht nach einem Link mit der Klasse `underline easy`, der explizit den Text "Standardsprache" enthält und auf die deutsche Version (`hreflang="de-DE"`) verweist.
7.  **Stadt Köln**
    *   **Strategie:** Suche nach einem spezifischen Hinweistext.
    *   **Details:** Es wird nach einem Link gesucht, dessen Text exakt "Diese Seite in Alltags-Sprache lesen" lautet (unabhängig von der Groß-/Kleinschreibung).
8.  **TAZ (taz.de)**
    *   **Strategie:** Extraktion aus einem hervorgehobenen Absatz.
    *   **Details:** Die taz platziert oft einen Hinweis auf den Originalartikel in einem kursiv gesetzten Absatz (`<em>`). Der Crawler sucht nach solchen Links innerhalb von Absätzen, um die Verknüpfung herzustellen.

### Zusammenfassung
Die meisten Crawler (Apotheken Umschau, MDR, TAZ etc.) nutzen explizite Rückverweise (Links) innerhalb der LS-Artikel, die zur Standardsprache führen. **Brand Eins** stellt eine Ausnahme dar, da hier die Trennung inhaltlich auf derselben Seite durch eine CSS-Farbcodierung erfolgt.

---

# Nächste Schritte

- [ ] **Quantifizierungs-Skript:** Entwicklung eines Skripts zum Scannen der identifizierten Quellen und zur Berechnung der aktuellen Token-Anzahlen.
- [ ] **Pipeline-Integration:** Einsatz dieser Zählskripte als „Pre-Scraper“, um die Datendichte vor der vollständigen Extraktion zu validieren.
- [ ] **Literatur & Erweiterung:** Systematische Überprüfung der verbleibenden Paper aus der Liste „Existing Corpora (Status 2023)“.
- [ ] **Quellenerweiterung:** Identifizierung zusätzlicher Quellen und Entwicklung seitenspezifischer Scraping-Mechanismen.
- [ ] **Crawler-Anpassung:** Adaption der Crawler-Logik von Toborek et al. an die neu identifizierten Strukturen (MDR, taz, etc.).

