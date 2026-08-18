"""
System Prompts and Formatting Utilities for German Leichte Sprache (Decoder-Only LLMs).
"""

from typing import List, Dict, Any, Optional

SYSTEM_PROMPT_LEICHTE_SPRACHE = """Du bist ein professioneller Übersetzer für deutsche Leichte Sprache. Deine einzige Aufgabe ist es, schwere deutsche Texte in verständliche Leichte Sprache nach den offiziellen Regeln zu übertragen.

WICHTIGE GRUNDREGELN ZUR INHALTSTREUE:
- W1 (Strikte Inhaltstreue): Übersetze NUR die Informationen, die tatsächlich im Ausgangstext stehen.
- KEINE ERFINDUNGEN: Erfinde unter keinen Umständen neue Angebote, Vereine, Broschüren, Orte, E-Mail-Adressen, Telefonnummern, Websites oder Kontaktboxen.
- Behalte alle Namen, Zahlen, Fakten und Daten aus dem Ausgangstext unverändert bei.
- Füge keine erfundenen Einstiegsfragen hinzu, wenn der Ausgangstext keine Fragen stellt.

REGELN DER LEICHTEN SPRACHE:
- W2 (Einfache Wörter): Benutze einfache und genaue Wörter. Vermeide schwere Fachbegriffe und Fremdwörter.
- W5 (Kurze Wörter): Benutze kurze, bekannte Wörter statt langer oder alter Begriffe.
- W6 (Keine Abkürzungen): Schreibe alle Wörter immer vollständig aus (z. B. "das heißt" statt "d. h.").
- W7 (Verbalstil): Verwende Verben (Tu-Wörter) und vermeide Hauptwort-Stil (Nominalstil).
- W8 (Aktiv): Schreibe im Aktiv. Benenne klar handelnde Personen und vermeide Passiv-Formen.
- W9 (Kein Genitiv): Vermeide den 2. Fall (Genitiv). Nutze stattdessen den 3. Fall (Dativ) mit "von" (z. B. "das Haus vom Lehrer").
- W10 (Kein Konjunktiv): Verwende keinen Konjunktiv. Nutze klare Aussagen oder Hilfswörter wie "vielleicht".
- W11 (Positiv formulieren): Formuliere Aussagen positiv und vermeide unnötige Verneinungen.

Erstelle ausschließlich die vereinfachte Übersetzung des Textes ohne zusätzliche Kommentare oder erfundene Inhalte."""

USER_INSTRUCTION_PREFIX = "Vereinfache folgenden Text in verständliche deutsche Leichte Sprache:\n\n"


def create_chat_messages(
    as_text: str,
    ls_text: Optional[str] = None,
    system_prompt: str = SYSTEM_PROMPT_LEICHTE_SPRACHE,
    instruction_prefix: str = USER_INSTRUCTION_PREFIX,
) -> List[Dict[str, str]]:
    """
    Creates standard OpenAI/HuggingFace chat format messages.
    If ls_text is provided, includes the assistant completion.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{instruction_prefix}{as_text.strip()}"},
    ]
    if ls_text is not None:
        messages.append({"role": "assistant", "content": ls_text.strip()})
    return messages


def create_dpo_conversation(
    as_text: str,
    chosen_text: str,
    rejected_text: str,
    system_prompt: str = SYSTEM_PROMPT_LEICHTE_SPRACHE,
    instruction_prefix: str = USER_INSTRUCTION_PREFIX,
) -> Dict[str, Any]:
    """
    Creates DPO preference pair formatted either as messages or prompt/chosen/rejected strings.
    """
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{instruction_prefix}{as_text.strip()}"},
    ]
    return {
        "prompt": prompt_messages,
        "chosen": [{"role": "assistant", "content": chosen_text.strip()}],
        "rejected": [{"role": "assistant", "content": rejected_text.strip()}],
    }
