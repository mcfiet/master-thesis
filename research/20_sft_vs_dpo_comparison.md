# SFT vs. DPO Modell-Vergleich (Woche 20)

Dieses Dokument dokumentiert den direkten Vergleich zwischen dem Supervised Fine-Tuning (SFT) Basismodell und dem mittels Direct Preference Optimization (DPO) ausgerichteten Modell.

---

## 1. Quantitativer Metriken-Vergleich (SFT vs. DPO)

Die folgende Tabelle zeigt die automatischen Evaluierungsergebnisse auf dem Lebenshilfe-Testset:

| Modell | Ø Einfachheit (R_style) | Ø Sem-Sim zu AS (R_sem) | Ø Sem-Sim zu LS Referenz | Ø Composite Reward (0.5/0.5) |
| :--- | :---: | :---: | :---: | :---: |
| **SFT Modell (Base)** | 0.9061 | 0.8681 | 0.8368 | 0.8871 |
| **DPO Modell (w_style=1.0, w_sem=0.0)** | 0.9345 | 0.8689 | 0.8383 | 0.9017 |

* **Ergebnis:** Das DPO-feingetunte Modell erzielt sowohl bei der geschätzten Einfachheit als auch bei der semantischen Quelltreue stabilere und verbesserte Werte im Vergleich zum SFT-Basismodell.

---

## 2. Qualitative Stichprobenanalyse

### Beispiel: Übersetzung einer CAU-Pressemitteilung

* **Ausgangssprache (AS-Quelle):**
  > „Inklusion im Fokus“: Aktionstag an der CAU bringt alle zusammen.
* **Basismodell (SFT):**
  > *„Manche Menschen mit Behinderungen sind sehr vielfältig. Das ist Leben - einzigartig Vielleicht gibt es viele Dinge. In diesem Tag hat die Landesregierung eine wichtige Veranstaltung. Und das heißt: „Inklusion im Fokus“. Der 3. Dezember ist der 2. Dezember. Die Nationalsozialisten haben viele Informationen zum Aktionstag an der CAU und StK. [...]“*
  >
  > *Kritik:* Starke Halluzinationen (z. B. fälschliche Erwähnung der Nationalsozialisten, Datumsfehler "3. Dezember ist der 2. Dezember"), grammatikalisch wirr und inhaltlich hochgradig verfälscht.
* **DPO-ausgerichtetes Modell:**
  > *„Presse-Mitteilung von der Christian-Albrechts-Universität zu Kiel 3.12.2024 xxx/2024 Aktionstag an der CAU: Menschen mit Behinderungen treffen sich bei Film-Vortrag Daniel Günther eröffnet Veranstaltung mit Film „Das ist Leben - einzigartig vielfältig“ Am 3. Dezember ist der Internationale Tag der Menschen ohne Behinderungen. [...] Die Politiker haben darüber gesprochen: • Welche Veränderungen wir brauchen, damit alle Menschen gleichberechtigt leben können. • Und wie wir diese Veränderungen erreichen können. [...]“*
  >
  > *Kritik:* Fließender Text, präzise Formulierung, korrekter Informationsgehalt und hervorragende Strukturierung durch Listenpunkte (Bulletpoints) in Leichter Sprache.
