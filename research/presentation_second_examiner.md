---
marp: true
theme: default
paginate: true
footer: "Master Thesis - Fiete Scheel"
style: |
  section { 
    font-family: 'Arial', sans-serif; 
    color: #555; 
    font-size: 24px;
    padding: 180px 40px 80px 40px; /* Increased Header and Footer Deadzones */
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
  }
  
  /* Logo Oben Rechts */
  section::before {
    content: '';
    position: absolute;
    top: 30px;
    right: 40px;
    width: 240px;
    height: 120px;
    background-image: url('img/presentation/hs_logo.png');
    background-size: contain;
    background-repeat: no-repeat;
    background-position: right top;
    z-index: 100;
  }

  /* Styling der Marp Seitenzahl + Footer Text (Unten Rechts) */
  section[data-marpit-pagination]::after {
    content: "Master Thesis - Fiete Scheel  |  " attr(data-marpit-pagination) " / " attr(data-marpit-pagination-total);
    position: absolute;
    bottom: 30px;
    right: 40px;
    font-size: 18px;
    color: #888;
  }

  /* Footer ausblenden (da wir ihn oben manuell in ::after rendern) */
  footer {
    display: none;
  }

  /* Dead Zone: Verhindert, dass Titel in das Logo fließen */
  h1, h2, h3 {
    color: #2c3e50;
  }

  h3 {
    position: absolute;
    top: 50px;
    left: 40px;
    width: calc(100% - 320px);
    font-size: 36px;
    margin: 0;
    line-height: 1.2;
  }

  /* Global Image Constraint: Ensure images never exceed the content area */
  section img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
    margin: 0 auto;
  }

  /* Reset für zentrierte Layouts, da diese mittig stehen sollen */
  section.title, section.section-header, section.big-number {
    padding: 100px 40px;
    justify-content: center;
  }

  table { font-size: 18px; }

  /* Layout: Titelfolie */
  section.title {
    text-align: center;
  }
  section.title h1 {
    font-size: 60px;
    margin-bottom: 20px;
  }

  /* Layout: Abschnittsüberschrift */
  section.section-header {
    background-color: #f4f7f6;
    text-align: center;
  }
  section.section-header h2 {
    font-size: 50px;
    display: inline-block;
    padding-bottom: 10px;
  }

  /* Layout: Zwei Spalten (Flexbox for better height control) */
  section.split {
    flex-direction: row !important; /* Force row layout */
    gap: 40px;
    align-items: stretch;
    justify-content: space-between;
    height: 100%;
    min-height: 0;
    box-sizing: border-box;
  }
  section.split > div {
    flex: 1;
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
  }

  /* Handle paragraphs in split layout */
  section.split div p {
    display: block;
    height: auto;
    margin: 0 0 20px 0; /* Default margin for text paragraphs */
  }

  /* Specialized handling for image-only paragraphs in split layout */
  section.split div p:has(img) {
    margin: 0;
    display: flex;
    justify-content: flex-start;
    height: 100%;
    min-height: 0;
  }

  section.split img {
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
    object-fit: contain;
  }

  /* Layout: Große Zahl */
  section.big-number {
    text-align: center;
  }
  section.big-number h1 {
    font-size: 120px;
    color: #2c3e50;
    margin: 0;
  }
  section.big-number p {
    font-size: 32px;
    font-weight: bold;
  }

  /* Layout: Bildunterschrift */
  section.image-caption {
    display: flex;
    flex-direction: column-reverse;
    justify-content: flex-start;
    align-items: flex-start;
    padding-top: 40px;
  }
  section.image-caption h3 {
    position: static;
    width: 100%;
    margin: 0;
    padding-top: 20px; /* Space between image and heading */
  }
  section.image-caption p {
    flex-grow: 1;
    min-height: 0; /* Allow p to shrink */
    display: flex;
    justify-content: flex-start; /* Align Left */
    align-items: flex-end; /* Align bottom */
    margin: 0;
    width: 100%;
    padding-right: 240px;
    box-sizing: border-box;
    overflow: hidden;
  }
  section.image-caption table {
    margin: 0;
    margin-right: 240px; /* Avoid logo */
    max-height: 100%;
    max-width: 100%;
    object-fit: contain;
  }
  section.image-caption img {
    max-height: 100%;
    max-width: 100%;
    object-fit: contain;
    margin: 0;
  }

  /* Layout: Kleiner, unauffälliger Hinweis (z.B. für Quellen oder Anmerkungen) */
  section .hint {
    font-size: 14px;
    color: #888;
    margin-top: 10px;
    font-style: italic;
    flex-grow: 0 !important;
    display: block !important;
  }

---

<!-- _class: title -->

# Master Thesis Update 

Entwicklung domänenspezifischer Datensätze und automatisierter Evaluation für ein Framework zur neuronalen Textvereinfachung in leichte Sprache

Fiete Scheel
Update | Juni 2026

---

### Motivation und Zielsetzung

**Barrierefreie Kommunikation:** Über 10 Mio. Menschen in Deutschland sind auf einfache oder leichte Sprache angewiesen.

**Problem:** Die manuelle Erstellung von "Leichter Sprache" (LS) ist zeitaufwendig, teuer und skaliert nicht.

**Kernfragen der Arbeit:**
1. Wie lässt sich ein qualitativ hochwertiger, domänenübergreifender **Parallelkorpus** (AS - LS) automatisiert aufbauen?
2. Mit welchen **Metriken** lässt sich die Qualität und der Informationsverlust objektiv messen?
3. Wie können **neuronale Sprachmodelle** trainiert werden, um regelkonforme Vereinfachungen zu generieren?
4. Lässt sich auf Basis des Datensatzes eine Metrik trainieren, um die Ergebnisse zu messen?

---

<!-- _class: section-header -->

## Korpus-Erstellung & Alignment 

---

### Aufbau des Parallelkorpus

Die größte Hürde für Machine Learning im Bereich "Leichte Sprache" ist der Mangel an strukturierten Daten.

**Vorgehensweise:**
- **Quellen Scraping:** Entwicklung maßgeschneiderter Crawler für News (MDR, taz), Gesundheit (Apotheken Umschau) und Verwaltung (Staatliche Portale).
- **Alignment-Strategien:** URL-Logik (z.B. Hannover), Sprach-Switch-Buttons (CSS-Patterns) und semantisches Matching.
- **Domänen-Abdeckung:** Nachrichten, Politik, Recht, Gesundheit und lokaler Bürgerservice.

---

### Aktuelle Korpus-Statistiken

| Quelle | Artikelpaare | Tokens (Standard) | Tokens (Leicht) |
| :--- | :---: | :---: | :---: |
| **Hannover** | 808 | ~871.000 | ~872.000 |
| **Apotheken Umschau** | 161 | ~451.000 | ~241.000 |
| **MDR** | 235 | ~173.000 | ~98.000 |
| **Hamburg / Stuttgart / Köln** | 194 | ~244.000 | ~171.000 |
| **Gesamt (Bereinigt)** | **1.471** | **~1.787.000** | **~1.432.000** |

**Ziel:** Einer der größten und am besten dokumentierten deutschsprachigen Parallelkorpora für Leichte Sprache. (Da es bisher kaum Forschung in diesem Gebiet gibt)

---

<!-- _class: section-header -->

## Qualitätssicherung & Analysis 

---

### Herausforderung: Informationsverlust messbar machen

**Theorie:** Leichte Sprache vereinfacht nicht nur, sie lässt Informationen weg (Reduktion auf den Kern).

**Analytische Ansätze:**
1. **Semantische Ähnlichkeit:** Messung über Embeddings.
   *   *Problem:* Herkömmliche Modelle (SBERT) schneiden bei 128/512 Tokens ab.
   *   *Lösung:* Einsatz von **Jina Embeddings v2** (8192 Tokens Kontext) für vollständige Artikel-Erfassung.
2. **NER Recall (Faktenerhalt):** Wieviele Eigennamen (Orte, Personen, Organisationen) überleben die Vereinfachung?
3. **Syntaktische Shifts:** Analyse von Satzlängen und Wortarten-Verteilung (POS).

---

<!-- _class: image-caption -->

### Analyse-Ergebnisse: Kontextlänge ist wichtig

![](img/analysis/jina_context_comparison.png)


---

<!-- _class: split -->

### Analyse-Ergebnisse: Struktur & Fakten

<div class="column-left">

**NER Recall (Bidirektional):**
Sowohl AS → LS als auch LS → AS zeigen niedrige Werte (~20-30%).
*   *Interpretation:* Die niedrigen Werte in **beide** Richtungen deuten darauf hin, dass Informationen nicht zwangsläufig verloren gehen, sondern **anders beschrieben** werden.
*   Eigennamen werden in LS oft paraphrasiert oder durch einfachere Begriffe ersetzt (z.B. "Arbeitsagentur" → "Amt für Arbeit").

</div>

<div class="column-right">

![](img/analysis/sentence_length_comparison_bar.png)

</div>

---

### Der "Gold Standard" Datensatz

Durch automatisierte Filterung wurde die Qualität für das Modelltraining maximiert:

**Filterkriterien:**
- **Ähnlichkeits-Korridor:** 0.60 < Score < 0.98 (entfernt Fehl-Alignments & Kopien).
- **Längen-Validierung:** Ausschluss von Teasern (< 10 Tokens).
- **Post-Cleaning:** Entfernung von Bildunterschriften, Radio-Metadaten und Navigations-Fragmenten.
- **Normalisierung:** Behandlung von Sonderzeichen wie dem Mediopunkt (`·`).

---

<!-- _class: section-header -->

## Nächste Schritte: Modellierung 

---

### Ausblick: Training & Evaluation

**1. Baseline Modelle:** 
Fine-Tuning von LLMs auf dem erstellten Gold-Standard Korpus.

**2. Reward-Modellierung:** 
Training einer Metrik als Belohnungsfunktion. Zusätzlich vlt. eine kombinierte Metrik (Ähnlichkeit + NER + trainiertes Modell).

**3. Qualitative Validierung:**
User-Studie oder Experten-Review (z.B. Lebenshilfe) zur Überprüfung der Lesbarkeit und inhaltlichen Korrektheit.

---

<!-- _class: title -->

# Vielen Dank!

**Fiete Scheel**
Fragen & Diskussion
