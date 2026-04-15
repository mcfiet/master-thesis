Lebenshilfe Kiel = 128 Worddokumente (64 LS, 64 normal)
https://sozialpolitik.com/es/seiten-uebersicht (25 Webseiten)

# Das Kernproblem

Es ist im Internet oft sehr schwierig, exakte Paralleltexte (wie Nachrichten oder Blogartikel) in Standardsprache (oder „schwerer Sprache“) und Leichter Sprache zu finden, die sich systematisch einander zuordnen lassen.

# Positivbeispiele
## sozialpolitik.com (Lösung über die URL)

Hier haben die Blogartikel in beiden Sprachversionen denselben Namen. Die Version in Leichter Sprache ist ganz einfach daran zu erkennen und aufzurufen, dass der URL lediglich das Präfix „ls-“ vorangestellt wird. Die URL des Blogposts ist unterschiedlich jedoch gibt es einen Button auf der jeweiligen Seite um zwischen "Leichte Sprache" und "Standardsprache" zu wechseln.
![[Pasted image 20260329181939.png]]
![[Pasted image 20260329182009.png]]

https://www.sozialpolitik.com/es/recht-auf-soziale-entschaedigung
https://www.sozialpolitik.com/opferentschaedigung


## mdr.de

Beim Mitteldeutschen Rundfunk funktioniert die Zuordnung zwar nicht über die URL, dafür ist die Benutzerführung innerhalb des Textes exzellent. Unter jedem Artikel in Leichter Sprache ist das Gegenstück in Standardsprache direkt verlinkt – und umgekehrt. So kann man nahtlos zwischen den Sprachniveaus hin- und herspringen. Nicht jeder Artikel in Standardsprache ist übersetzt in LS aber jeder Artikel in LS ist übersetzt in Standardsprache (stichprobenartig kontrolliert).

![[Pasted image 20260329182048.png]]
![[Pasted image 20260329182105.png]]


# Negativbeispiele

## Tagesschau

Die Tagesschau fällt bei der Verknüpfung von Standard- und Leichter Sprache leider negativ auf, da es an einer nutzerfreundlichen und systematischen Architektur mangelt:
- **Fehlende Verlinkung:** Es gibt in den Artikeln keinen Button oder direkten Link, um einfach zwischen den verschiedenen Sprachniveaus hin- und herzuwechseln.
- **Inkonsistente URL-Struktur:** Die grundlegende Anlage der URLs wirkt zwar durchdacht, in der Praxis sind die Artikel jedoch unterschiedlich benannt. Es gibt kein verlässliches Muster (wie ein festes Präfix), wodurch eine systematische oder gar automatisierte Zuordnung der Texte unmöglich wird.
- **Hoher manueller Aufwand:** Um Paralleltexte zu finden, bleibt nur die aufwendige händische Suche nach dem semantisch passenden Gegenstück.
- **Starke inhaltliche Reduktion:** Selbst wenn man das entsprechende Artikel-Paar gefunden hat, gestaltet sich ein direkter Textvergleich schwierig, da die Version in Leichter Sprache inhaltlich massiv gekürzt und stark vereinfacht ist.

https://www.tagesschau.de/inland/bundestagswahl/leichte-sprache/bundestagswahl-in-leichter-sprache-148.html
https://www.tagesschau.de/inland/bundestagswahl/bundestags-aussteiger-100.html

# Nächste Schritte

- [ ] **Quantifizierung der Quellen:** Ermittlung der ungefähren Anzahl an verfügbaren Artikel-Paaren bei mdr.de und sozialpolitik.com.
- [ ] **Scraper-Entwicklung:** Erstellung eines Prototyps zum automatisierten Auslesen der sozialpolitik.com-Artikel (Nutzung der URL-Logik).
- [ ] **MDR-Crawler:** Untersuchung, ob die Verlinkungen zwischen LS und Standardsprache bei mdr.de systematisch gecrawlt werden können.
- [ ] **Matching-Strategie für Tagesschau:** Evaluation, ob semantische Ähnlichkeitsanalysen (z. B. via Embeddings) genutzt werden können, um die fehlenden Verknüpfungen bei der Tagesschau automatisiert herzustellen.
- [ ] **Recherche weiterer Quellen:** Suche nach weiteren Nachrichtenportalen oder offiziellen Regierungsseiten (z. B. bpb.de), die systematisch LS-Texte anbieten.