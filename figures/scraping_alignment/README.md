# Scraping Alignment Visualisierungs-Report

Dieses Verzeichnis enthält hochauflösende Screenshots aller 12 Korpusquellen zur visuellen Erklärung des Ausrichtungs- und Extraktionsprozesses (Alignment zwischen Leichter Sprache und Standardsprache).

## Übersicht der Quellen & Alignment-Strategien

| Quelle | Kategorie | Alignment-Mechanismus | Screenshots |
| :--- | :--- | :--- | :--- |
| **Apotheken Umschau** | Healthcare / Medical | In-Text Cross-Reference Link | [Zu den Bildern](#apotheken) |
| **Behindertenbeauftragter** | Federal Government / Official | Header-Servicenavigation Sprachwechsler | [Zu den Bildern](#behindertenbeauftragter) |
| **brand eins** | Journalism / Economy (Archive) | In-Page Parallele Textblöcke & Farbcodierung | [Zu den Bildern](#brandeins) |
| **Hamburg.de** | Municipality / Portal | Dedizierte Language-Bar ('Originaltext') | [Zu den Bildern](#hamburg) |
| **Hannover.de** | Municipality / Portal | Button 'Zur Seite in Alltagssprache' & Canonical Link | [Zu den Bildern](#hannover) |
| **Stadt Köln** | Municipality / Portal (Archive) | In-Content Switch Link ('Alltags-Sprache lesen') | [Zu den Bildern](#koeln) |
| **Lebenshilfe Main-Taunus** | NGO / Inclusion (Archive) | Header-Sprachumschalter mit Tooltip/Icon | [Zu den Bildern](#main_taunus) |
| **MDR (Mitteldeutscher Rundfunk)** | Public Broadcasting / News | Teaser-Box 'In schwerer Sprache lesen' | [Zu den Bildern](#mdr) |
| **Sozialpolitik.com** | Educational / Federal Ministry (BMAS) | Header Quick-Switch 'Standardsprache' | [Zu den Bildern](#sozialpolitik) |
| **Stuttgart.de** | Municipality / Portal | Content-Button 'Artikel in Alltags-Sprache' / URL-Parameter | [Zu den Bildern](#stuttgart) |
| **taz (die tageszeitung)** | Journalism / News | Redaktioneller In-Text Quellverweis ('schwerer Text') | [Zu den Bildern](#taz) |
| **Wiesbaden.de** | Municipality / Portal | Sprachleisten-Toggle 'Leichte Sprache' / 'Alltagssprache' | [Zu den Bildern](#wiesbaden) |

---

### Apotheken Umschau <a id='apotheken'></a>
**Strategie**: `In-Text Cross-Reference Link`  
**Beschreibung**: Am Ende des Leichte-Sprache-Artikels verlinkt die Apotheken Umschau mit Texten wie 'Hier finden Sie noch mehr Informationen über...' oder 'Informationen' auf den standardsprachlichen Fachartikel.  
- **LS URL**: [https://www.apotheken-umschau.de/einfache-sprache/krankheiten/weitsichtigkeit-753261.html](https://www.apotheken-umschau.de/einfache-sprache/krankheiten/weitsichtigkeit-753261.html)
- **AS URL**: [https://www.apotheken-umschau.de/krankheiten-symptome/augenkrankheiten/wie-kann-man-weitsichtigkeit-behandeln-733661.html](https://www.apotheken-umschau.de/krankheiten-symptome/augenkrankheiten/wie-kann-man-weitsichtigkeit-behandeln-733661.html)

#### 1. Leichte Sprache mit hervorgehobenem Alignment-Element
![Apotheken Umschau LS Viewport](apotheken/apotheken_viewport.png)

#### 2. Detailansicht des Alignment-Elements
![Apotheken Umschau Closeup](apotheken/apotheken_element_closeup.png)

#### 3. Standardsprachlicher Gegenstück-Artikel (AS)
![Apotheken Umschau AS Target](apotheken/apotheken_as_target.png)

#### 4. Paarweiser Vergleich (LS vs. AS)
![Apotheken Umschau Gegenüberstellung](apotheken/apotheken_side_by_side.png)

---

### Behindertenbeauftragter <a id='behindertenbeauftragter'></a>
**Strategie**: `Header-Servicenavigation Sprachwechsler`  
**Beschreibung**: In der oberen Metanavigation existiert ein Sprachumschalter (navServiceAS / Alltagssprache-Icon), der den direkten Wechsel zur Alltagssprache ermöglicht.  
- **LS URL**: [https://www.behindertenbeauftragter.de/DE/LS/presse-und-aktuelles/veranstaltungen/sonderseiten/BRKKonferenz/FactSheet_07.html](https://www.behindertenbeauftragter.de/DE/LS/presse-und-aktuelles/veranstaltungen/sonderseiten/BRKKonferenz/FactSheet_07.html)
- **AS URL**: [https://www.behindertenbeauftragter.de/DE/AS/startseite/startseite-node.html](https://www.behindertenbeauftragter.de/DE/AS/startseite/startseite-node.html)

#### 1. Leichte Sprache mit hervorgehobenem Alignment-Element
![Behindertenbeauftragter LS Viewport](behindertenbeauftragter/behindertenbeauftragter_viewport.png)

#### 3. Standardsprachlicher Gegenstück-Artikel (AS)
![Behindertenbeauftragter AS Target](behindertenbeauftragter/behindertenbeauftragter_as_target.png)

#### 4. Paarweiser Vergleich (LS vs. AS)
![Behindertenbeauftragter Gegenüberstellung](behindertenbeauftragter/behindertenbeauftragter_side_by_side.png)

---

### brand eins <a id='brandeins'></a>
**Strategie**: `In-Page Parallele Textblöcke & Farbcodierung`  
**Beschreibung**: brand eins stellt LS- und AS-Versionen direkt im selben Artikel gegenüber: Absätze in Alltagssprache (blau umrahmt) werden durch farbcodierte Absätze in Leichter Sprache (grün umrahmt) paarweise ergänzt.  
- **LS URL**: [https://web.archive.org/web/20240528075943/https://www.brandeins.de/magazine/brand-eins-wirtschaftsmagazin/2022/abo-wirtschaft/leichte-sprache-hauptsache-es-stand-in-irgendeiner-liste](https://web.archive.org/web/20240528075943/https://www.brandeins.de/magazine/brand-eins-wirtschaftsmagazin/2022/abo-wirtschaft/leichte-sprache-hauptsache-es-stand-in-irgendeiner-liste)

#### 1. Leichte Sprache mit hervorgehobenem Alignment-Element
![brand eins LS Viewport](brandeins/brandeins_viewport.png)

#### 2. Detailansicht des Alignment-Elements
![brand eins Closeup](brandeins/brandeins_element_closeup.png)

---

### Hamburg.de <a id='hamburg'></a>
**Strategie**: `Dedizierte Language-Bar ('Originaltext')`  
**Beschreibung**: Jedem Leichte-Sprache-Artikel ist eine strukturierte Sprachleiste (.km1-language-bar) mit dem Button 'Originaltext / Alltagssprache' vorangestellt.  
- **LS URL**: [https://www.hamburg.de/barrierefrei/leichte-sprache/polizei-feuerwehr/ls-starkregen-1019916](https://www.hamburg.de/barrierefrei/leichte-sprache/polizei-feuerwehr/ls-starkregen-1019916)
- **AS URL**: [https://www.hamburg.de/politik-und-verwaltung/behoerden/bukea/themen/klima/starkregen-946792](https://www.hamburg.de/politik-und-verwaltung/behoerden/bukea/themen/klima/starkregen-946792)

#### 1. Leichte Sprache mit hervorgehobenem Alignment-Element
![Hamburg.de LS Viewport](hamburg/hamburg_viewport.png)

#### 2. Detailansicht des Alignment-Elements
![Hamburg.de Closeup](hamburg/hamburg_element_closeup.png)

#### 3. Standardsprachlicher Gegenstück-Artikel (AS)
![Hamburg.de AS Target](hamburg/hamburg_as_target.png)

#### 4. Paarweiser Vergleich (LS vs. AS)
![Hamburg.de Gegenüberstellung](hamburg/hamburg_side_by_side.png)

---

### Hannover.de <a id='hannover'></a>
**Strategie**: `Button 'Zur Seite in Alltagssprache' & Canonical Link`  
**Beschreibung**: Hannover.de platziert am Artikelanfang einen auffälligen Button 'Zur Seite in Alltagssprache' (.schwer.icon) und referenziert die AS-Version im Canonical Link.  
- **LS URL**: [https://www.hannover.de/Leichte-Sprache/Hannover-und-Region/Politik/Wahlen/Kommunal∙wahlen-2026-in-der-Region-Hannover](https://www.hannover.de/Leichte-Sprache/Hannover-und-Region/Politik/Wahlen/Kommunal∙wahlen-2026-in-der-Region-Hannover)
- **AS URL**: [https://www.hannover.de/Leben-in-der-Region-Hannover/Politik/Wahlen-Statistik/Kommunalwahlen-2026-in-der-Region-Hannover](https://www.hannover.de/Leben-in-der-Region-Hannover/Politik/Wahlen-Statistik/Kommunalwahlen-2026-in-der-Region-Hannover)

#### 1. Leichte Sprache mit hervorgehobenem Alignment-Element
![Hannover.de LS Viewport](hannover/hannover_viewport.png)

#### 2. Detailansicht des Alignment-Elements
![Hannover.de Closeup](hannover/hannover_element_closeup.png)

#### 3. Standardsprachlicher Gegenstück-Artikel (AS)
![Hannover.de AS Target](hannover/hannover_as_target.png)

#### 4. Paarweiser Vergleich (LS vs. AS)
![Hannover.de Gegenüberstellung](hannover/hannover_side_by_side.png)

---

### Stadt Köln <a id='koeln'></a>
**Strategie**: `In-Content Switch Link ('Alltags-Sprache lesen')`  
**Beschreibung**: Im Hauptinhalt jedes LS-Dokuments führt der Navigationslink 'Alltags-Sprache lesen' direkt zur regulären Dienstleistungsseite.  
- **LS URL**: [https://web.archive.org/web/20220804230818/https://www.stadt-koeln.de/leben-in-koeln/soziales/unterhalts-vorschuss](https://web.archive.org/web/20220804230818/https://www.stadt-koeln.de/leben-in-koeln/soziales/unterhalts-vorschuss)
- **AS URL**: [https://web.archive.org/web/20220401090142/https://www.stadt-koeln.de/service/produkt/unterhaltsvorschuss-1](https://web.archive.org/web/20220401090142/https://www.stadt-koeln.de/service/produkt/unterhaltsvorschuss-1)

#### 1. Leichte Sprache mit hervorgehobenem Alignment-Element
![Stadt Köln LS Viewport](koeln/koeln_viewport.png)

#### 2. Detailansicht des Alignment-Elements
![Stadt Köln Closeup](koeln/koeln_element_closeup.png)

#### 3. Standardsprachlicher Gegenstück-Artikel (AS)
![Stadt Köln AS Target](koeln/koeln_as_target.png)

#### 4. Paarweiser Vergleich (LS vs. AS)
![Stadt Köln Gegenüberstellung](koeln/koeln_side_by_side.png)

---

### Lebenshilfe Main-Taunus <a id='main_taunus'></a>
**Strategie**: `Header-Sprachumschalter mit Tooltip/Icon`  
**Beschreibung**: Ein verankerter Header-Button mit dem Attribut title='Auf Alltags-Sprache umstellen' und dem Text 'Alltags-Sprache' dient als Umschaltpunkt.  
- **LS URL**: [https://web.archive.org/web/20210122220843/https://www.lebenshilfe-main-taunus.de/ls/reisen.html](https://web.archive.org/web/20210122220843/https://www.lebenshilfe-main-taunus.de/ls/reisen.html)
- **AS URL**: [https://web.archive.org/web/20210225194538/https://www.lebenshilfe-main-taunus.de/reisen.html](https://web.archive.org/web/20210225194538/https://www.lebenshilfe-main-taunus.de/reisen.html)

#### 1. Leichte Sprache mit hervorgehobenem Alignment-Element
![Lebenshilfe Main-Taunus LS Viewport](main_taunus/main_taunus_viewport.png)

#### 2. Detailansicht des Alignment-Elements
![Lebenshilfe Main-Taunus Closeup](main_taunus/main_taunus_element_closeup.png)

#### 3. Standardsprachlicher Gegenstück-Artikel (AS)
![Lebenshilfe Main-Taunus AS Target](main_taunus/main_taunus_as_target.png)

#### 4. Paarweiser Vergleich (LS vs. AS)
![Lebenshilfe Main-Taunus Gegenüberstellung](main_taunus/main_taunus_side_by_side.png)

---

### MDR (Mitteldeutscher Rundfunk) <a id='mdr'></a>
**Strategie**: `Teaser-Box 'In schwerer Sprache lesen'`  
**Beschreibung**: Unterhalb des LS-Textes bindet der MDR eine Teaser-Box mit der Überschrift 'HIER KÖNNEN SIE DIESE NACHRICHT AUCH IN SCHWERER SPRACHE LESEN:' ein, die zum redaktionellen Fachartikel führt.  
- **LS URL**: [https://www.mdr.de/nachrichten-leicht/leichte-sprache-sachsen-sachsenforst-waldbrand-saison-100.html](https://www.mdr.de/nachrichten-leicht/leichte-sprache-sachsen-sachsenforst-waldbrand-saison-100.html)
- **AS URL**: [https://www.mdr.de/nachrichten/sachsen/wetter-fruehling-wochenende-waldbrand-gefahr-102.html](https://www.mdr.de/nachrichten/sachsen/wetter-fruehling-wochenende-waldbrand-gefahr-102.html)

#### 1. Leichte Sprache mit hervorgehobenem Alignment-Element
![MDR (Mitteldeutscher Rundfunk) LS Viewport](mdr/mdr_viewport.png)

#### 2. Detailansicht des Alignment-Elements
![MDR (Mitteldeutscher Rundfunk) Closeup](mdr/mdr_element_closeup.png)

#### 3. Standardsprachlicher Gegenstück-Artikel (AS)
![MDR (Mitteldeutscher Rundfunk) AS Target](mdr/mdr_as_target.png)

#### 4. Paarweiser Vergleich (LS vs. AS)
![MDR (Mitteldeutscher Rundfunk) Gegenüberstellung](mdr/mdr_side_by_side.png)

---

### Sozialpolitik.com <a id='sozialpolitik'></a>
**Strategie**: `Header Quick-Switch 'Standardsprache'`  
**Beschreibung**: In der oberen Leiste ermöglicht der Button 'Standardsprache' (.underline.easy) das direkte Hin- und Herschalten zur Standardfassung der Bildungseinheit.  
- **LS URL**: [https://www.sozialpolitik.com/es/auswirkungen-der-coronavirus-epidemie](https://www.sozialpolitik.com/es/auswirkungen-der-coronavirus-epidemie)
- **AS URL**: [https://www.sozialpolitik.com/auswirkungen-der-coronavirus-epidemie](https://www.sozialpolitik.com/auswirkungen-der-coronavirus-epidemie)

#### 1. Leichte Sprache mit hervorgehobenem Alignment-Element
![Sozialpolitik.com LS Viewport](sozialpolitik/sozialpolitik_viewport.png)

#### 2. Detailansicht des Alignment-Elements
![Sozialpolitik.com Closeup](sozialpolitik/sozialpolitik_element_closeup.png)

#### 3. Standardsprachlicher Gegenstück-Artikel (AS)
![Sozialpolitik.com AS Target](sozialpolitik/sozialpolitik_as_target.png)

#### 4. Paarweiser Vergleich (LS vs. AS)
![Sozialpolitik.com Gegenüberstellung](sozialpolitik/sozialpolitik_side_by_side.png)

---

### Stuttgart.de <a id='stuttgart'></a>
**Strategie**: `Content-Button 'Artikel in Alltags-Sprache' / URL-Parameter`  
**Beschreibung**: Stuttgart.de verwendet einen Aktionsbutton 'Artikel in Alltags-Sprache' (.SP-Link) und steuert die Barrierefreiheit über den URL-Parameter ?sp:out=easy.  
- **LS URL**: [https://www.stuttgart.de/leben/gesundheit/vorsorge/suchtpraevention?sp%3Aout=easy](https://www.stuttgart.de/leben/gesundheit/vorsorge/suchtpraevention?sp%3Aout=easy)
- **AS URL**: [https://www.stuttgart.de/leben/gesundheit/vorsorge/suchtpraevention](https://www.stuttgart.de/leben/gesundheit/vorsorge/suchtpraevention)

#### 1. Leichte Sprache mit hervorgehobenem Alignment-Element
![Stuttgart.de LS Viewport](stuttgart/stuttgart_viewport.png)

#### 2. Detailansicht des Alignment-Elements
![Stuttgart.de Closeup](stuttgart/stuttgart_element_closeup.png)

#### 3. Standardsprachlicher Gegenstück-Artikel (AS)
![Stuttgart.de AS Target](stuttgart/stuttgart_as_target.png)

#### 4. Paarweiser Vergleich (LS vs. AS)
![Stuttgart.de Gegenüberstellung](stuttgart/stuttgart_side_by_side.png)

---

### taz (die tageszeitung) <a id='taz'></a>
**Strategie**: `Redaktioneller In-Text Quellverweis ('schwerer Text')`  
**Beschreibung**: Die taz-Redaktion verlinkt im Text mit redaktionellen Hinweisen wie 'aus diesem „schweren“ Text' auf den originalen Hintergrundartikel.  
- **LS URL**: [https://taz.de/Leichte-Sprache/!5590875/](https://taz.de/Leichte-Sprache/!5590875/)
- **AS URL**: [https://taz.de/Wahlzulassung-fuer-Betreute/!5588713/](https://taz.de/Wahlzulassung-fuer-Betreute/!5588713/)

#### 1. Leichte Sprache mit hervorgehobenem Alignment-Element
![taz (die tageszeitung) LS Viewport](taz/taz_viewport.png)

#### 2. Detailansicht des Alignment-Elements
![taz (die tageszeitung) Closeup](taz/taz_element_closeup.png)

#### 3. Standardsprachlicher Gegenstück-Artikel (AS)
![taz (die tageszeitung) AS Target](taz/taz_as_target.png)

#### 4. Paarweiser Vergleich (LS vs. AS)
![taz (die tageszeitung) Gegenüberstellung](taz/taz_side_by_side.png)

---

### Wiesbaden.de <a id='wiesbaden'></a>
**Strategie**: `Sprachleisten-Toggle 'Leichte Sprache' / 'Alltagssprache'`  
**Beschreibung**: Wiesbaden.de besitzt in der Funktionsleiste einen Umschaltlink (.SP-Link--simple-language) mit Parameter ?sp:easylanguage=1, der direkt zwischen Normalfassung und barrierefreier Fassung wechselt.  
- **LS URL**: [https://www.wiesbaden.de/vv/produkte/31/Fuehrerschein-umtauschen-online-beantragen?sp%3Aeasylanguage=1](https://www.wiesbaden.de/vv/produkte/31/Fuehrerschein-umtauschen-online-beantragen?sp%3Aeasylanguage=1)
- **AS URL**: [https://www.wiesbaden.de/vv/produkte/31/Fuehrerschein-umtauschen-online-beantragen](https://www.wiesbaden.de/vv/produkte/31/Fuehrerschein-umtauschen-online-beantragen)

#### 1. Leichte Sprache mit hervorgehobenem Alignment-Element
![Wiesbaden.de LS Viewport](wiesbaden/wiesbaden_viewport.png)

#### 2. Detailansicht des Alignment-Elements
![Wiesbaden.de Closeup](wiesbaden/wiesbaden_element_closeup.png)

#### 3. Standardsprachlicher Gegenstück-Artikel (AS)
![Wiesbaden.de AS Target](wiesbaden/wiesbaden_as_target.png)

#### 4. Paarweiser Vergleich (LS vs. AS)
![Wiesbaden.de Gegenüberstellung](wiesbaden/wiesbaden_side_by_side.png)

---
