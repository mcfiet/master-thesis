# Web-Applikation: Next.js & DPO-Modell-Integration (Woche 20)

Dieses Dokument dokumentiert die technische Zusammenführung der Web-Komponenten, die Initialisierung des Next.js-Projekts sowie die Einbindung des feingetunten DPO-Übersetzungsmodells.

---

## 1. Restrukturierung & Next.js-Initialisierung

Um die Codebasis übersichtlicher zu gestalten, wurden das Frontend und das Backend in einem gemeinsamen Verzeichnis zusammengeführt:

* **Backend:** Die Flask-API, welche die Modellgenerierung steuert und die API-Endpunkte bereitstellt, liegt nun direkt neben den Frontend-Assets.
* **Frontend:** Initialisierung eines modernen Next.js-Projekts im selben Ordner, um eine interaktive Benutzeroberfläche zur Live-Übersetzung von Alltagssprache in Leichte Sprache bereitzustellen.

---

## 2. Integration des DPO-Übersetzungsmodells

Die Web-Applikation greift nun direkt auf das verbesserte DPO-Übersetzungsmodell auf Basis von `mbart-large-50` zu:

* **Modell-Integration:** Die Flask-API lädt das DPO-Modell, das mit unserer Verbund-Reward-Funktion (Style-Reward und Semantik-Score) trainiert wurde.
* **Standardisierung der Ausgabe:** Die Übersetzungslogik und die API wurden auf die vereinheitlichte Skala angepasst:
  * `1.0` = Maximale Einfachheit (Zielgrad Leichte Sprache erreicht).
  * `0.0` = Reiner Ausgangstext (Alltagssprache).
  * Dies ermöglicht es der Weboberfläche, dem Nutzer visuelles Feedback über den geschätzten Vereinfachungsgrad des übersetzten Texts zurückzugeben.

---

## 3. Lokale Ausführung

* **Starten des Dev-Servers:**
  * Backend und Frontend werden parallel über separate Konsolen gestartet, um eine latenzarme Echtzeit-Generierung zu ermöglichen.
* Die Anwendung ermöglicht eine Echtzeit-Übersetzung mit einstellbaren Generierungsparametern (wie Repetition Penalty und Search-Beam-Breite) über das Next.js Frontend.
