#!/usr/bin/env python3
"""
Generate intermediate synthetic regression steps between Leichte Sprache (LS) and Alltagssprache (AS)
using an HTTP-accessible OpenAI-compatible chat completion endpoint.

Reads a JSON file with paired (AS, LS) articles (e.g., lebenshilfe_dataset.json),
prompts the model with both versions to generate texts at intermediate levels of complexity
(e.g., 0.25, 0.50, 0.75), and saves the results to a new JSON file.

Supports incremental saving and resuming so already generated steps are not lost.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any

import requests

SYSTEM_PROMPT = """Du bist ein Experte für die deutsche Sprache und Textvereinfachung.
Dir wird ein Artikel in zwei verschiedenen Schwierigkeitsgraden vorgelegt:
1. Leichte Sprache (LS / Stufe 0.0): Extrem vereinfacht, kurze Sätze, sehr einfache Satzstruktur, oft mit Zeilenumbrüchen nach jedem Satzteil, geringe Wortkomplexität, keine Fremdwörter, Fachbegriffe werden direkt erklärt.
2. Alltagssprache (AS / Stufe 1.0): Der normale Originaltext, komplexer Satzbau, reichhaltiger Wortschatz, Nebensätze, Passiv- und Genitivkonstruktionen, Fachbegriffe.

Deine Aufgabe ist es, eine neue Version dieses Artikels zu schreiben, die sprachlich präzise auf einer bestimmten Stufe zwischen diesen beiden Versionen liegt.
Die Skala reicht von 0.0 (Leichte Sprache) bis 1.0 (Alltagssprache).
Dir wird ein bestimmter Zielwert (z.B. 0.25, 0.50, 0.75) vorgegeben.

Richtlinien für die Stufen:
- Stufe 0.25 (Nahe an Leichter Sprache):
  - Der Text soll sehr leicht verständlich sein.
  - Verwende überwiegend einfache Sätze, aber du darfst einzelne einfache Nebensätze (z. B. mit "weil", "wenn", "dass") einbauen.
  - Der Wortschatz darf minimal anspruchsvoller sein als in der LS-Version (z. B. geläufige Wörter anstelle von extremen Umschreibungen), aber vermeide Fremdwörter.
  - Informationen sollten sehr klar gegliedert sein, aber es muss nicht jeder Satzteil in einer neuen Zeile stehen.

- Stufe 0.50 (Die goldene Mitte / Einfache Sprache):
  - Dies entspricht der typischen "Einfachen Sprache" (Simple Language).
  - Verwende eine Mischung aus kurzen und mittelschweren Sätzen.
  - Vermeide Schachtelsätze (mehrere Nebensätze ineinander), aber ein Hauptsatz mit einem Nebensatz ist der Standard.
  - Der Wortschatz ist alltäglich, ohne extreme Fach- oder Fremdwörter.
  - Der Text fließt natürlicher als Leichte Sprache, behält aber die gute Verständlichkeit bei.

- Stufe 0.75 (Nahe an Alltagssprache):
  - Der Text ist nur leicht vereinfacht im Vergleich zur AS-Version.
  - Schachtelsätze werden weitgehend in zwei separate Sätze aufgeteilt.
  - Schwierige Fachbegriffe oder Fremdwörter werden entweder vermieden, durch geläufigere Wörter ersetzt oder im Satz kurz erklärt.
  - Der Satzbau ist flüssig und entspricht einem gut lesbaren Zeitungsartikel.

Wichtige Regeln:
1. Verändere den Inhalt nicht: Es dürfen keine Fakten oder Informationen hinzuerfunden oder weggelassen werden, die nicht in den beiden Quelltexten vorhanden sind.
2. Orientiere dich stilistisch und inhaltlich an den vorgegebenen Versionen.
3. Gib ausschließlich den neu generierten Text zurück. Keine Metakommentare, keine Erklärungen, keine Anmerkungen, keine Markdown-Codeblöcke. Beginne direkt mit dem Text des Artikels.
"""

def build_prompt(ls_text: str, as_text: str, target_level: float) -> str:
    return (
        f"Hier sind die beiden Versionen des Artikels:\n\n"
        f"### Version 1: Leichte Sprache (LS / Stufe 0.0):\n{ls_text.strip()}\n\n"
        f"### Version 2: Alltagssprache (AS / Stufe 1.0):\n{as_text.strip()}\n\n"
        f"### Ziel-Stufe: {target_level:.2f}\n\n"
        f"Erstelle nun den Text, der genau auf der Ziel-Stufe {target_level:.2f} liegt. "
        f"Befolge alle stilistischen Richtlinien für diese Stufe. Antworte nur mit dem generierten Text."
    )

def clean_response(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

def call_model(
    url: str,
    prompt: str,
    system: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    model: str | None = None,
    token: str | None = None,
) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model:
        payload["model"] = model

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected response schema: {json.dumps(data)[:500]}") from exc
    return clean_response(content)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate intermediate synthetic articles between LS and AS using an HTTP model endpoint."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/lebenshilfe_dataset.json"),
        help="Input JSON file containing paired LS and AS articles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/lebenshilfe_dataset_with_steps.json"),
        help="Output JSON file with intermediate generated steps.",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="HTTP endpoint of the model (OpenAI-compatible chat completion URL).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name to include in the payload.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional API token (Bearer token) for authentication.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="0.25,0.50,0.75",
        help="Comma-separated float values representing intermediate steps (between 0.0 and 1.0).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate per article step.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of input articles to process.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep in seconds between requests to avoid overloading the server.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resuming from an existing output file (overwrites it completely).",
    )
    return parser.parse_args()

def get_key(item: Dict[str, Any]) -> tuple:
    return (item.get("ls_filename"), item.get("as_filename"))

def load_existing_results(path: Path) -> Dict[tuple, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {get_key(item): item for item in data if get_key(item)[0] is not None}
    except Exception as e:
        print(f"[warn] Could not parse existing output file {path}: {e}. Starting fresh.")
        return {}

def save_results(results_map: Dict[tuple, Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    results_list = list(results_map.values())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)

def main():
    args = parse_args()

    # Parse target steps
    target_steps = []
    try:
        for s in args.steps.split(","):
            val = float(s.strip())
            if not (0.0 < val < 1.0):
                raise ValueError(f"Step {val} is not between 0.0 and 1.0 (exclusive)")
            target_steps.append(val)
    except Exception as exc:
        print(f"[error] Failed to parse steps '{args.steps}': {exc}")
        return

    # Load input dataset
    if not args.input.exists():
        print(f"[error] Input file {args.input} does not exist.")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    if args.limit:
        input_data = input_data[:args.limit]

    # Load existing progress if resuming
    results_map = {}
    if not args.no_resume:
        results_map = load_existing_results(args.output)
        print(f"[info] Loaded {len(results_map)} existing articles from output file.")

    total_articles = len(input_data)
    generated_count = 0

    print(f"[info] Processing {total_articles} articles for steps: {target_steps}")

    for idx, item in enumerate(input_data, 1):
        ls_text = item.get("ls_text", "")
        as_text = item.get("as_text", "")
        key = get_key(item)

        # Check if already processed
        existing_item = results_map.get(key)
        if existing_item is None:
            # Initialize entry
            existing_item = item.copy()
            if "intermediate_steps" not in existing_item:
                existing_item["intermediate_steps"] = {}
            results_map[key] = existing_item

        intermediate_steps = existing_item.setdefault("intermediate_steps", {})

        # Determine which steps still need generation
        steps_to_generate = [step for step in target_steps if f"{step:.2f}" not in intermediate_steps]

        if not steps_to_generate:
            print(f"[info] [{idx}/{total_articles}] Skipping '{key[0] or 'unknown'}', all steps already exist.")
            continue

        print(f"[info] [{idx}/{total_articles}] Generating steps {steps_to_generate} for '{key[0] or 'unknown'}'...")

        success = True
        for step in steps_to_generate:
            step_key = f"{step:.2f}"
            prompt = build_prompt(ls_text, as_text, step)
            
            try:
                print(f"  -> Calling model for step {step_key}...")
                start_time = time.time()
                generated_text = call_model(
                    url=args.url,
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    model=args.model,
                    token=args.token,
                )
                elapsed = time.time() - start_time
                print(f"  -> Success ({elapsed:.1f}s, {len(generated_text)} chars)")
                intermediate_steps[step_key] = generated_text
                generated_count += 1
                
                # Save after each successful step to minimize data loss risk
                save_results(results_map, args.output)

                if args.sleep > 0:
                    time.sleep(args.sleep)

            except Exception as exc:
                print(f"[warn] Failed to generate step {step_key} for '{key[0]}': {exc}")
                success = False
                # If a step fails, we stop generating further steps for this article for now
                break

        if success:
            print(f"[info] [{idx}/{total_articles}] Completed all steps for '{key[0]}'.")

    print(f"[done] Processed {total_articles} articles. Generated {generated_count} new steps. Output saved to {args.output}")

if __name__ == "__main__":
    main()
