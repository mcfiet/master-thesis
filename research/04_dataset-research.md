# Übersicht: Alignierte Korpora (Standarddeutsch <-> Leichte/Einfache Sprache)

Basierend auf der Analyse der Textpassage (u.a. Klaper et al., Battisti et al.) wurden folgende Korpora identifiziert, die Standarddeutsch und vereinfachte Versionen (Leichte Sprache/Einfache Sprache) parallel oder aligniert gegenüberstellen:

### 1. Satz-alignierte Korpora (Sentence-aligned)
- **Klaper et al. [38]**: Das erste parallele, satz-alignierte Korpus für Deutsch und Leichte Sprache (ca. 70.000 Token). 
  - [Link (ACL Anthology)](https://aclanthology.org/W13-2902/)
- **Säuberli et al. [56] (APA-Korpus)**: Fokus auf Automatic Text Simplification (ATS); enthält 3.616 manuell alignierte Satzpaare (Original vs. A2/B1).
  - [Link (ACL Anthology)](https://aclanthology.org/2020.readi-1.7/)
- **Hansen-Schirra et al. [32] (Geasy)**: Professionelle Übersetzungen in Leichte Sprache, auf Satzebene aligniert. Enthält 1.087.643 Wörter Quelltext und 292.552 Wörter Leichte Sprache. *Hinweis: Das Verhältnis von Ziel- zu Ausgangstext liegt bei nur ca. 27%, was auf starke inhaltliche Kürzungen hindeutet und die Anzahl perfekter 1:1 Satz-Alignments verringern dürfte.*
  - [Link (Springer)](https://link.springer.com/chapter/10.1007/978-981-16-4918-9_11) | [Projektseite](https://traco.uni-mainz.de/geasy-korpus/)
- **Toborek et al. [62]**: Enthält 5.942 alignierte Sätze (sowie 708 alignierte Dokumente) aus verschiedenen News-Quellen.
  - [Link (ACL Anthology)](https://aclanthology.org/2023.acl-long.638/)
- **Naderi et al. [49] (TextComplexityDE)**: Enthält 250 manuell vereinfachte Sätze (aus einem Pool von 1.000), inklusive subjektiver Komplexitätsbewertungen.
  - [Link (arXiv)](https://arxiv.org/abs/1904.07733) | [GitHub](https://github.com/babaknaderi/TextComplexityDE)

### 2. Dokument-alignierte / Parallele Korpora
- **Battisti et al. [8]**: Parallele Daten (Deutsch/Leichte Sprache) mit ca. 348k Token (Standard) und 246k Token (Leichte Sprache). Enthält auch Struktur- und Metadaten (Typografie/Bilder).
  - [Link (ACL Anthology)](https://aclanthology.org/2020.lrec-1.404/) | [Zenodo (Daten)](https://zenodo.org/record/6576356)
- **Spring et al. [59]**: Erweiterung des APA-Korpus; bietet 2.410 Dokumentpaare für B1 und 2.347 für A2 (Zeitraum 2018–2021).
  - [Link (ACL Anthology - Verwandte Arbeit)](https://aclanthology.org/2021.gem-1.18/)
- **Aumiller und Gertz [5] (Klexicon)**: Alignment auf Dokumentebene zwischen Wikipedia-Artikeln und dem Kinderlexikon *Klexikon* (2.898 Artikelpaare).
  - [Link (ACL Anthology)](https://aclanthology.org/2022.lrec-1.282/)

### 3. Wichtige Abgrenzungen (Nicht oder nur teilweise aligniert)
- **Rios et al. [55] (20m)**: 18.305 Artikelpaare (News vs. Zusammenfassung), aber **nicht** auf Satzebene aligniert.
  - [Link (ACL Anthology)](https://aclanthology.org/2021.gem-1.18/)
- **Hauser et al. [33] (SNIML)**: Multilinguales News-Korpus; Alignment zum Standarddeutschen ist erst für zukünftige Versionen geplant.
  - [Link (arXiv)](https://arxiv.org/abs/2205.11142)
- **Jablotschkin und Zinsmeister [35] (LeiKo)**: Wird als "comparable corpus" (vergleichbar) bezeichnet, nicht zwingend strikt parallel.
  - [Link (ACL Anthology)](https://aclanthology.org/2022.lrec-1.579/)
- **Lange [39] (LeiSa)**: Projekt "Leichte Sprache im Arbeitsleben"; Fokus auf vergleichende Analyse.
  - [Link (Verlag/Projekt)](https://www.frank-timme.de/de/programm/produkt/leichte-sprache-im-arbeitsleben)
- **Jach [36] (KED)**: Umfangreiche Sammlung (Korpus Einfaches Deutsch), eher monolingual/vergleichbar als strikt parallel.
  - [Link (Projektseite)](https://korpus-einfaches-deutsch.de/)