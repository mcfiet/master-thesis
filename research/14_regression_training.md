Regression

- Variante 1: Mix-Up (Satz von gleicher Länge jeweils von LS und AS)
- Variante 2: Mit LLM und LS und AS Texts als Input zwischen Steps

---

## Umsetzung von Variante 2: LLM-basierte Generierung von Zwischenstufen

Für den Regressionsansatz haben wir ein Setup aufgebaut, das für jedes Artikelpaar (LS und AS) synthetische Zwischenschritte der sprachlichen Komplexität generiert.

### 1. Generierungsskript (generate_synthetic_regression_steps.py)

Es wurde das Skript generate_synthetic_regression_steps.py erstellt. Dieses basiert auf dem ursprünglichen HTTP-Skript aus dem `genai-project` und wurde an unseren Artikel-Anwendungsfall angepasst:

- **Eingabe:** lebenshilfe_dataset.json (49 verifizierte AS-LS-Artikelpaare).
- **Prompting:** Übergibt dem LLM die Originalversion (AS / Stufe 1.0) und die einfache Version (LS / Stufe 0.0) und weist das Modell an, einen Text zu generieren, der exakt auf einer gewünschten Ziel-Stufe liegt (z. B. 0.25, 0.50, 0.75).
- **Stufendefinitionen im System-Prompt:**
  - **Stufe 0.25 (Nahe an Leichter Sprache):** Sehr leicht verständlich, einfache Sätze, vereinzelte einfache Nebensätze (z. B. mit "weil", "wenn"), keine Fremdwörter, strukturierter Fließtext.
  - **Stufe 0.50 (Die goldene Mitte / Einfache Sprache):** Mischung aus kurzen und mittelschweren Sätzen, keine Schachtelsätze, alltäglicher Wortschatz, flüssiger Text.
  - **Stufe 0.75 (Nahe an Alltagssprache):** Leicht vereinfachte Alltagssprache, Schachtelsätze werden geteilt, Fremdwörter/Fachbegriffe umschrieben oder im Satz kurz erklärt.
- **Ausgabe:** Schreibt die Ergebnisse inkrementell nach lebenshilfe_dataset_with_steps.json.
- **Robustheit (Resume-Funktion):** Bereits generierte Stufen werden bei einem Neustart erkannt und übersprungen.

### 2. Ausführung & Test

Das Skript wird über die virtuelle Python-Umgebung ausgeführt.

#### Lokaler Testlauf (mit Ollama):

1. **Ollama-App** auf dem Mac starten (stellt die API bereit).
2. Modell pullen:
   ```bash
   ollama pull llama3
   ```
3. Testlauf mit 1 Artikel durchführen:
   ```bash
   .venv/bin/python scripts/generate_synthetic_regression_steps.py --url http://localhost:11434/v1/chat/completions --model llama3 --limit 1
   ```
4. Gesamten Durchlauf starten:
   ```bash
   .venv/bin/python scripts/generate_synthetic_regression_steps.py --url http://localhost:11434/v1/chat/completions --model llama3
   ```

#### Remote-Ausführung (auf dem GPU-Server):

Für die Ausführung mit dem großen Modell `FlensGen-GPT-OSS120B` auf dem Server:

```bash
.venv/bin/python scripts/generate_synthetic_regression_steps.py --url http://193.175.188.202:8000/v1/chat/completions --model "FlensGen-GPT-OSS120B"
```

_Hinweis:_ Hierfür muss die VPN-Verbindung (Cisco AnyConnect) aktiv sein, da die IP-Adresse `193.175.188.202` im Hochschulnetzwerk liegt und von außen geblockt wird.

---

## Ergebnisse des Testlaufs (Ollama / LLaMA 3)

Wir haben das Skript für einen Testlauf gestartet. Das Skript hat erfolgreich die drei Stufen `0.25`, `0.50` und `0.75` generiert und in [lebenshilfe_dataset_with_steps.json](file:///Users/fietescheel/Documents/Master%20Thesis/data/lebenshilfe/lebenshilfe_dataset_with_steps.json) eingetragen.

### Auszug der generierten Stufen für den ersten Artikel:

- **Stufe 0.25 (Nahe an Leichter Sprache):**
  _"Hier ist der Text in Leichter Sprache auf der Ziel-Stufe 0.25: Inklusion im Fokus: Aktionstag an der CAU bringt alle zusammen..."_
- **Stufe 0.50 (Die goldene Mitte):**
  _"Hier ist der Text in Leichter Sprache auf der Ziel-Stufe 0.50: Inklusion im Fokus: Aktionstag an der CAU bringt alle zusammen..."_
- **Stufe 0.75 (Nahe an Alltagssprache):**
  _"Hier ist der Text in leichter Sprache auf der Ziel-Stufe 0.75: Inklusion im Fokus: Aktionstag an der CAU bringt alle zusammen..."_

---

## Probleme & Verbesserungspotenziale (Lessons Learned)

Bei der Auswertung des ersten Testlaufs sind folgende Punkte aufgefallen:

1. **Mismatched Dataset-Paare in den Quelldaten (Kritisch):**
   - _Beobachtung:_ Der erste Artikel im Datensatz verknüpft fälschlicherweise das Dokument `ILS_CAU_Geologiemuseum` (Leichte Sprache) mit der Pressemitteilung `20241106-PM-Aktionstag-CAU und StK` (Alltagssprache).
   - _Problem:_ Das LLM erhält zwei völlig unterschiedliche Themen und entscheidet sich meist dafür, nur das AS-Thema (Inklusion) zu vereinfachen, wodurch der LS-Vergleichstext ignoriert wird.
   - _Lösung:_ Die manuellen Zuordnungen in `scripts/create_lebenshilfe_dataset.py` (Zeile 67ff) müssen bereinigt und korrigiert werden.

2. **Einleitende Floskeln des LLMs filtern:**
   - _Beobachtung:_ Das LLM fügt trotz gegenteiliger Instruktion am Anfang des Texts Sätze wie _"Hier ist der Text in Leichter Sprache auf der Ziel-Stufe 0.25:"_ hinzu.
   - _Lösung:_ Das Postprocessing im Skript sollte erweitert werden, um solche typischen Höflichkeits- und Einleitungsphrasen automatisch zu entfernen (z. B. wenn eine Zeile mit `"Hier ist"` oder `"Generierter"` beginnt).

3. **Layout- & Formatierungserhalt bei Stufe 0.25:**
   - _Beobachtung:_ Leichte Sprache lebt von Aufzählungspunkten und kurzen Zeilen. Das LLM hat bei Stufe `0.25` den Text in normalen Fließtext umgewandelt.
   - _Lösung:_ Der Systemprompt kann bezüglich der Strukturierung verfeinert werden (z. B. _"Erhalte Aufzählungszeichen und zeilenbasierte Strukturierung auf niedrigen Stufen wie 0.25, falls die LS-Hintergrundversion diese nutzt"_).

4. **Automatische Validierung (Readability Metrics):**
   - _Lösung:_ Um zu verifizieren, ob die Stufen tatsächlich eine Komplexitäts-Hierarchie abbilden, sollte ein kurzes Skript geschrieben werden, das den Flesch-Index und die Wiener Sachtextformel für jede generierte Stufe berechnet und visualisiert.

---

## Konzeptuelle Umsetzung von Variante 1: Mix-Up

Für die Umsetzung von **Variante 1** (Mix-Up) auf Satz- oder Embedding-Ebene gibt es drei verschiedene methodische Ansätze, die wir ausprobieren können:

### Ansatz A: Embedding-Space Interpolation (Mathematischer Mix-Up)

Dieser Ansatz arbeitet direkt auf Vektorebene und eignet sich hervorragend für Regressionen auf Embeddings (z. B. mit einem MLP-Kopf wie in `mlp_training.ipynb`).

- **Funktionsweise:** Wir berechnen die SBERT-Embeddings für den Alltagssatz ($E_{AS}$) und den dazugehörigen einfachen Satz ($E_{LS}$). Wir interpolieren diese beiden Vektoren linear über ein Gewicht $\lambda \in [0.0, 1.0]$:
  $$E_{mix} = \lambda \cdot E_{LS} + (1 - \lambda) \cdot E_{AS}$$
- **Vorteil:** Benötigt keine LLM-Generierung und erzeugt kontinuierliche Übergänge.
- **Regressionstarget:** Der Zielwert für das Modell ist direkt $\lambda$ (wobei $0.0 = AS$ und $1.0 = LS$).

### Ansatz B: Satz-Level Mix-Up (Diskreter Absatz-Mix)

Dieser Ansatz mischt Sätze auf Absatz- oder Artikelebene und benötigt ebenfalls kein LLM.

- **Funktionsweise:** Wir erstellen künstliche Artikel/Absätze, indem wir Sätze aus der AS-Version und der LS-Version mischen. Bei einem Absatz mit $N$ Sätzen definieren wir die Mischverhältnisse:
  - **Stufe 0.00 (AS):** Nur AS-Sätze ($N$ AS, $0$ LS)
  - **Stufe 0.25:** Mischung aus $75\,\%$ AS-Sätzen und $25\,\%$ LS-Sätzen
  - **Stufe 0.50:** Mischung aus $50\,\%$ AS-Sätzen und $50\,\%$ LS-Sätzen
  - **Stufe 0.75:** Mischung aus $25\,\%$ AS-Sätzen und $75\,\%$ LS-Sätzen
  - **Stufe 1.00 (LS):** Nur LS-Sätze ($0$ AS, $N$ LS)
- **Vorteil:** Erzeugt echten, lesbaren Text ohne Generierungskosten.

### Ansatz C: Token-Level Mix-Up (Token-Substitution bei gleicher Länge)

Dieser Ansatz arbeitet auf Wortebene mit syntaktisch ähnlichen Sätzen.

- **Funktionsweise:** Wir suchen gezielt nach Satzpaaren (AS/LS), die die gleiche Wort- oder Token-Länge $L$ aufweisen (oder bringen sie durch Padding auf dieselbe Länge). Wir erzeugen Zwischenstufen, indem wir an zufälligen Positionen $i$ Wörter der AS-Version durch die Wörter an Position $i$ der LS-Version ersetzen.
- **Vorteil:** Erlaubt die Untersuchung feingranularer syntaktischer Übergangsgrenzen direkt auf Token-Ebene.

## Feedback

### MixUp Variante 1

- Im besten Fall sollte man einen Satz Alltagssprache und dann z.B. drei Saetze in leichter Sprache hintendran damit diese den AS-Satz weiterfuehren. Da das aber nicht so moeglich ist, macht man die Anordnung einfach random.
- Darauf achten im Dataloader dass die Saetze gleich verteilt sind
- 100% und 100% mischen sollte gehen; Model ist ja gegen Lange robust
