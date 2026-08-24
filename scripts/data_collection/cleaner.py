#!/usr/bin/env python3
"""
Zentrales Reinigungs- und Normalisierungsmodul für das Korpus-Scraping
und Preprocessing in Leichter Sprache (LS) und Alltagssprache (AS).

Behebt:
1. Fehlende Satzzeichen an Block-/Tag-Grenzen (Tag-Punctuation-Guard).
2. Führende Kicker / Ortsmarken (z. B. 'Sachsen', 'Sachsen-Anhalt', 'Thüringen').
3. Listen-Artefakte & Bullet-Points ('•', '*', ': •', ' - ') in flüssigen Text.
4. Doppelte Satzzeichen ('..', ': ..', ', ,', '..;').
5. Mediopunkte ('·', '∙') und fehlende Leerzeichen nach Punkten.
6. Web-Navigation, Autoren-Credits, Prüfer-Signaturen & Boilerplate.
"""

import re

MONTHS = r"(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)"

def ensure_block_punctuation(block_text: str) -> str:
    """
    Stellt sicher, dass ein HTML-Textblock (h1-h6, p, li, div) mit einem gültigen
    Satzzeichen (. ! ? :) endet, bevor Blöcke zusammengefügt werden.
    Verhindert das Verschmelzen von Sätzen ohne Satzzeichen.
    """
    if not block_text:
        return ""
    text = block_text.strip()
    if not text:
        return ""
    if text[-1] in ".!?:":
        return text
    if text[-1] in ",;":
        return text[:-1] + "."
    return text + "."

def clean_header_kicker(text: str) -> str:
    """
    Entfernt isolierte Bundesland-Kicker und Dachzeilen am Textanfang (MDR etc.).
    Beispiel: 'Sachsen Jetzt beginnt...' -> 'Jetzt beginnt...'
    """
    if not text:
        return ""
    kicker_pattern = r'^(SACHSEN-ANHALT|Sachsen-Anhalt|SACHSEN|Sachsen|THÜRINGEN|Thüringen|MITTEL-DEUTSCHLAND|Mittel-Deutschland|Mitteldeutschland|in Leichter Sprache Ticker:?|Glossar)\s*[:.\-]?\s+'
    text = re.sub(kicker_pattern, '', text, flags=re.IGNORECASE).strip()
    return text

def convert_bullets_and_lists(text: str) -> str:
    """
    Transformiert Bullet-Points ('•'), Asterisks ('*') und Aufzählungspunkte
    in grammatikalisch korrekten, flüssigen Fließtext.
    """
    if not text:
        return ""

    # 1. Spezifische Köln-Asterisk Bildnachweis-Boilerplate komplett entfernen
    text = re.sub(r'\*\s*Die Bilder gehören:.*$', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'Weitere Infos\s+\*\s+.*$', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'Weitere Informationen\s+\*\s+.*$', '', text, flags=re.IGNORECASE | re.DOTALL)

    # 2. Entfernung von ': •' -> ': ' (Doppelpunkt vor Liste beibehalten, Bullet entfernen)
    text = re.sub(r':\s*•\s*', ': ', text)

    # 3. Bullet nach Satzende (. •, ! •, ? •) -> Satzzeichen bleibt, Bullet weg
    text = re.sub(r'([.!?])\s*•\s*', r'\1 ', text)

    # 4. Bullet vor Konjunktionen (• und, • oder, • sowie, • aber) -> Konjunktion verbindet
    text = re.sub(r'\s*•\s*(und|oder|aber|sowie|wie)\b', r' \1', text, flags=re.IGNORECASE)

    # 5. Bullet zwischen Aufzählungselementen (Nomen/Verben) -> Komma-Separation
    text = re.sub(r'\s*•\s*', ', ', text)

    # 6. Bereinigung von Überbleibseln
    text = re.sub(r':\s*,\s*', ': ', text)          # ': ,' -> ': '
    text = re.sub(r',\s+(und|oder|aber|sowie)\b', r' \1', text, flags=re.IGNORECASE) # ', und' -> ' und'
    text = re.sub(r',\s*,', ',', text)              # doppelte Kommas
    text = re.sub(r'\*\s*', '', text)               # verbleibende Asterisks entfernen

    return text

def normalize_typography(text: str) -> str:
    """
    Bereinigt typographische Artefakte, Mediopunkte, Mehrfachpunkte und Abstände.
    """
    if not text:
        return ""

    # 1. Mediopunkte entfernen
    text = text.replace('·', '').replace('∙', '')

    # 2. Mehrfache Punkte und Satzzeichenkombinationen
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r':\s*\.+', ':', text)            # ': ..' -> ':'
    text = re.sub(r';\s*\.+', ';', text)            # '; ..' -> ';'
    text = re.sub(r'\.\s*;\s*', '. ', text)         # '..;' -> '. '

    # 3. Fehlende Leerzeichen nach Punkten vor Großbuchstaben einfügen
    text = re.sub(r'([a-zäöüß0-9])\.([A-ZÄÖÜ])', r'\1. \2', text)

    # 4. Leerzeichen vor Satzzeichen entfernen
    text = re.sub(r'\s+([?.!,;:])', r'\1', text)

    # 5. Doppelte Leerzeichen normalisieren
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def clean_source_specific(text: str, source: str) -> str:
    """
    Entfernt quellenspezifische Signaturen, Autorenzeilen und Boilerplate.
    """
    if not text:
        return ""

    source = (source or "").lower()

    if source == "brandeins":
        text = re.sub(rf'^{MONTHS} \d{{4}}\.', '', text).strip()
        text = re.sub(rf'^.*? {MONTHS} \d{{4}}\.[A-Z][a-z]+(\s[A-Z][a-z]+)?', '', text).strip()
        text = re.sub(rf'^.*? {MONTHS} \d{{4}}\.', '', text).strip()

    elif source == "mdr":
        text = re.sub(r'Über dieses Thema berichtet der MDR auch in schwerer Sprache:.*?$', '', text, flags=re.DOTALL | re.IGNORECASE)

    elif source == "taz":
        text = re.sub(r'Das ist [A-ZÄÖÜ][a-z]+(\s[A-ZÄÖÜ][a-z]+)? vor seinem Laden:(\s)?', '', text)
        text = re.sub(r'Übertragung in Leichte Sprache von:.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Prüfung von:.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Erschienen am:\s*\d+\.\s*[A-Za-z]+\s+\d{4}', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Die Infos in diesem leichten Text kommen aus.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)

    elif source == "hamburg":
        text = re.sub(r'.*?haben den Text geschrieben und gelesen.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'.*?haben den Text geprüft.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'.*?hat die Bilder gemalt.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Der Text ist geschrieben und geprüft nach den Regeln von.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Der Text ist vom Büro für Leichte Sprache.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)

    elif source == "apotheken":
        text = re.sub(r'Welche Frage zu.*?Unser Tool durchsucht unsere Artikel.*?(\s\w+)?$', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'MEHR ANZEIGEN\s+\w+$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Die Texte haben wir zusammen mit der Forschungsstelle Leichte Sprache geschrieben.*?Universität Hildesheim(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Wo bekommen Sie noch mehr Informationen\?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Hier finden Sie mehr Informationen über.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Achtung\s*:\s*Dieser Link führt aus unserem Einfache-Sprache-Angebot heraus.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Die Informationen sind dann nicht mehr in Einfacher Sprache.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Sie wollen noch mehr über.*?lesen\?.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Achtung:\s*In diesem Text finden Sie nur allgemeine Informationen.*?Rufen Sie in der Arztpraxis an(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Wichtig:\s*Sie möchten Heilpflanzen gegen Ihre Beschwerden nehmen.*?In der Apotheke erfahren Sie.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)

    elif source == "hannover":
        text = re.sub(r'Sie interessieren sich für ein bestimmtes Thema\?\s*Dann klicken Sie auf ein Feld\.?', '', text)
        text = re.sub(r'Klicken Sie hier.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Hier finden Sie.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Mehr Informationen in Alltagssprache.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Icon für die Mobilversion.*?(\.|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Regionspräsident Steffen Krach.*?Oberbürgermeister.*?(\.|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Von der [A-Za-zÄÖÜäöüß\-]+straße aus fotografiert:.*?(\.|$)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Das Team vom [A-Za-z0-9\-]+-Infobus', '', text, flags=re.IGNORECASE)

    elif source in ["stuttgart", "koeln"]:
        text = re.sub(r'Die Bilder im Text sind von.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Illustrator Stefan Albers.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Atelier Fleetinsel.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'© European Easy-to-Read Logo.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Mehr Informationen im Internet unter.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Internetseite von Inclusion Europe.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)

    return text.strip()

def remove_web_navigation(text: str) -> str:
    """
    Entfernt allgemeine Web-Navigation, Suchwerkzeuge, Verlinkungen und Call-to-Actions.
    """
    if not text:
        return ""

    # 1. Navigations-Fragen
    text = re.sub(r'Wo (finde|bekomme) ich (noch |weitere |mehr )?(Informationen|Infos).*?\?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Sie möchten (noch |weitere |mehr )?(Informationen|Infos).*?\?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Sie wollen noch mehr (über|zu).*?lesen\??', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Sie interessieren sich für.*?\?(\s*Dann klicken Sie.*?\.)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Welche Frage zu.*?haben Sie\?\s*Unser Tool durchsucht unsere Artikel.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Ihre Frage wird nicht gespeichert\.\s*(Augen|[A-Za-z]+)?', '', text, flags=re.IGNORECASE)

    # 2. Verweis-Sätze
    text = re.sub(r'Hier (erfahren|lesen|bekommen|finden) Sie (mehr|alles|weitere|etwas).*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Erfahren|Lesen) Sie mehr (über|zu|zum Thema).*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Mehr (Informationen|Infos) (über|zu|in Alltagssprache|im Internet).*?(auf dieser Seite|hier|finden Sie|erfahren Sie|lesen Sie|unter).*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Mehr Infos (zu|zur|zum)\s+[\w\-]+:?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Hier finden Sie (weitere )?Infos.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Hier kommen Sie (zum|zur|zu).*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Sprechen Sie uns an!?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Auf dieser Seite finden Sie (viele |mehr |weitere )?Informationen zum Thema:.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Weitere (Informationen|Infos) (in Alltagssprache|zum Thema|über).*?(\.|$)', '', text, flags=re.IGNORECASE)

    # 3. Klick-Anweisungen & Linklisten
    text = re.sub(r'(\(Alltagssprache\)|\(in Alltagssprache\))', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Einen Link (für|zu).*?(unten|hier).*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Links?\s+(Link\s+)?(zum|zur|zu|unter).*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(\(Wenn Sie.*?online lesen möchten,\s*\))?([kK]licken|[tT]ippen) Sie (bitte )?(auf|hier).*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Dann finden Sie in dieser Tabelle weitere Informationen.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Achtung:\s*Dieser Link führt.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Die Informationen sind dann nicht mehr in (Einfacher|Leichter) Sprache.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Hier geht es zu.*?(\.|$)', '', text, flags=re.IGNORECASE)

    # 4. URLs und Protokollreste (z. B. 'https://... (Abgerufen am...)')
    text = re.sub(r'https?://\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Abgerufen am \d{2}\.\d{2}\.\d{4}\)', '', text, flags=re.IGNORECASE)

    return text.strip()

def clean_text(text: str, source: str = "") -> str:
    """
    Führt die vollständige Bereinigungskette für einen einzelnen Text aus.
    """
    if not text:
        return ""

    # 1. Kicker / Dachzeilen am Anfang entfernen
    text = clean_header_kicker(text)

    # 2. Bullet-Points & Listen transformieren
    text = convert_bullets_and_lists(text)

    # 3. Quellenspezifische Bereinigung
    if source:
        text = clean_source_specific(text, source)

    # 4. Web-Navigation & Boilerplate entfernen
    text = remove_web_navigation(text)

    # 5. Typographie und Satzzeichen normalisieren
    text = normalize_typography(text)

    return text

def clean_pair(ls_text: str, as_text: str, source: str = "") -> tuple:
    """
    Bereinigt ein (LS, AS)-Textpaar vollständig.
    """
    clean_ls = clean_text(ls_text, source)
    clean_as = clean_text(as_text, source)
    return clean_ls, clean_as
