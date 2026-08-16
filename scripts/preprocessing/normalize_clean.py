import json
import os
import re
from tqdm import tqdm

# Configuration (Passed via Command Line)
import argparse
parser = argparse.ArgumentParser(description="Normalize, deduplicate and filter corpus text pairs.")
parser.add_argument("--input_dir", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--min_ratio", type=float, default=0.20, help="Minimum LS/AS word length ratio (default: 0.20)")
parser.add_argument("--max_ratio", type=float, default=4.00, help="Maximum LS/AS word length ratio (default: 4.00)")
parser.add_argument("--min_words", type=int, default=30, help="Minimum word count per text (default: 30)")
args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
MIN_RATIO = args.min_ratio
MAX_RATIO = args.max_ratio
MIN_WORDS = args.min_words

def is_valid_length_ratio(ls_text: str, as_text: str, min_ratio: float = 0.20, max_ratio: float = 4.00, min_words: int = 30) -> bool:
    """
    Filters out extreme length ratio outliers and texts with insufficient word count.
    Prevents pairings of multi-page manuals with short teaser stubs.
    """
    ls_len = len(ls_text.split())
    as_len = len(as_text.split())
    if ls_len < min_words or as_len < min_words:
        return False
    ratio = ls_len / as_len
    return min_ratio <= ratio <= max_ratio

MONTHS = r"(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)"

def clean_brandeins(text):
    """
    Cleans Brand Eins specific artifacts:
    - Metadata like 'März 2023.' or 'Mai 2021.' at the beginning.
    - Missing spaces after periods.
    """
    # 1. Remove metadata at the beginning: [Month] [Year].
    # Pattern: ^[Some text maybe]? [Month] [Year]. [Maybe some name]
    # We target the Month Year dot pattern specifically.
    text = re.sub(rf'^{MONTHS} \d{{4}}\.', '', text).strip()
    
    # Sometimes the title is also there, separated by a dot.
    # Pattern: ^.*? [Month] [Year]\.[Name]?
    # Example: "Sie werden mit falschem Käse betrogen März 2023.Holger Fr Parmesan..."
    text = re.sub(rf'^.*? {MONTHS} \d{{4}}\.[A-Z][a-z]+(\s[A-Z][a-z]+)?', '', text).strip()
    # Case without name
    text = re.sub(rf'^.*? {MONTHS} \d{{4}}\.', '', text).strip()

    # 2. Add missing space after period followed by uppercase letter
    text = re.sub(r'\.([A-ZÄÖÜ])', r'. \1', text)
    
    return text

def clean_mdr(text):
    """
    Removes MDR boilerplate footers.
    """
    # Pattern: "Über dieses Thema berichtet der MDR auch in schwerer Sprache: ..."
    text = re.sub(r'Über dieses Thema berichtet der MDR auch in schwerer Sprache:.*?$', '', text, flags=re.DOTALL)
    return text.strip()

def clean_taz(text):
    """
    Cleans TAZ specific artifacts.
    """
    # Remove orphaned image captions like "Das ist Baris Kul vor seinem Laden:"
    text = re.sub(r'Das ist [A-ZÄÖÜ][a-z]+(\s[A-ZÄÖÜ][a-z]+)? vor seinem Laden:(\s)?', '', text)
    return text.strip()

def clean_apotheken(text):
    """
    Removes typical apotheken website boilerplate & Hildesheim university signatures.
    Only targets specific footer boilerplate structures to avoid over-cleaning.
    """
    # Remove search tool & question boilerplates: "Welche Frage zu... Unser Tool durchsucht unsere Artikel..."
    text = re.sub(r'Welche Frage zu.*?Unser Tool durchsucht unsere Artikel.*?(\s\w+)?$', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'MEHR ANZEIGEN\s+\w+$', '', text, flags=re.IGNORECASE)
    
    # Remove Hildesheim Forschungsstelle credits (only as full credits paragraph, not single words)
    text = re.sub(r'Die Texte haben wir zusammen mit der Forschungsstelle Leichte Sprache geschrieben.*?Universität Hildesheim(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove the specific link warning block (only when it contains the link warning context)
    text = re.sub(r'Wo bekommen Sie noch mehr Informationen\?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Hier finden Sie mehr Informationen über.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Achtung\s*:\s*Dieser Link führt aus unserem Einfache-Sprache-Angebot heraus.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Die Informationen sind dann nicht mehr in Einfacher Sprache.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Sie wollen noch mehr über.*?lesen\?.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove standard physician consultation advice (only as a complete disclaimer paragraph)
    text = re.sub(r'Achtung:\s*In diesem Text finden Sie nur allgemeine Informationen.*?Rufen Sie in der Arztpraxis an(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Wichtig:\s*Sie möchten Heilpflanzen gegen Ihre Beschwerden nehmen.*?In der Apotheke erfahren Sie.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Clean double whitespaces and strip
    return re.sub(r'\s+', ' ', text).strip()

def clean_taz_hamburg_credits(text):
    """
    Removes specific translator, writer, and checker credits (e.g. from TAZ leicht or Hamburg).
    """
    # Remove TAZ translator signature
    text = re.sub(r'Übertragung in Leichte Sprache von:.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Prüfung von:.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Erschienen am:\s*\d+\.\s*[A-Za-z]+\s+\d{4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Die Infos in diesem leichten Text kommen aus.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove Hamburg / Lisi GmbH credits
    text = re.sub(r'.*?haben den Text geschrieben und gelesen.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'.*?haben den Text geprüft.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'.*?hat die Bilder gemalt.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Der Text ist geschrieben und geprüft nach den Regeln von.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Der Text ist vom Büro für Leichte Sprache.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)

    return re.sub(r'\s+', ' ', text).strip()

def clean_hannover(text):
    """
    Removes Hanover boilerplate lines, image captions, and icon artifacts.
    """
    # Remove specific theme selection bias phrase
    text = re.sub(r'Sie interessieren sich für ein bestimmtes Thema\?\s*Dann klicken Sie auf ein Feld\.?', '', text)
    
    # Remove 'Klicken Sie' links and references
    text = re.sub(r'Klicken Sie hier.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Hier finden Sie.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Mehr Informationen in Alltagssprache.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Achtung:.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove orphaned image caption / icon text artifacts
    text = re.sub(r'Icon für die Mobilversion.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Regionspräsident Steffen Krach.*?Oberbürgermeister.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Von der [A-Za-zÄÖÜäöüß\-]+straße aus fotografiert:.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Das Team vom [A-Za-z0-9\-]+-Infobus', '', text, flags=re.IGNORECASE)
    
    return re.sub(r'\s+', ' ', text).strip()

def clean_stuttgart_koeln(text):
    """
    Removes signature credits, illustrators and translators from Stuttgart and Cologne articles.
    """
    # Remove Stefan Albers illustrator signatures
    text = re.sub(r'Die Bilder im Text sind von.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Illustrator Stefan Albers.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Atelier Fleetinsel.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove Inclusion Europe logo information
    text = re.sub(r'© European Easy-to-Read Logo.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Mehr Informationen im Internet unter.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'Internetseite von Inclusion Europe.*?(\.|$)', '', text, flags=re.DOTALL | re.IGNORECASE)

    return re.sub(r'\s+', ' ', text).strip()

def clean_stuttgart(text):
    """
    Cleans Stuttgart specific artifacts like repeated titles.
    """
    # First apply general illustrator signatures
    text = clean_stuttgart_koeln(text)
    return text

def clean_navigation_boilerplate(text):
    """
    Removes cross-source web navigation, link referrals, and SEO search artifacts.
    """
    # 1. Question headers asking where to find more info / how to navigate
    text = re.sub(r'Wo (finde|bekomme) ich (noch |weitere |mehr )?(Informationen|Infos).*?\?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Sie möchten (noch |weitere |mehr )?(Informationen|Infos).*?\?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Sie wollen noch mehr (über|zu).*?lesen\??', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Sie interessieren sich für.*?\?(\s*Dann klicken Sie.*?\.)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Welche Frage zu.*?haben Sie\?\s*Unser Tool durchsucht unsere Artikel.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Ihre Frage wird nicht gespeichert\.\s*(Augen|[A-Za-z]+)?', '', text, flags=re.IGNORECASE)
    
    # 2. Sentences referring to more info / links on this page or external sites
    text = re.sub(r'Mehr Informationen (über|zu|in Alltagssprache|im Internet).*?(auf dieser Seite|hier|finden Sie|unter).*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Hier (finden|bekommen) Sie (mehr|weitere) Informationen.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Hier finden Sie (weitere )?Infos.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Auf dieser Seite finden Sie (viele |mehr |weitere )?Informationen zum Thema:.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Weitere Informationen (in Alltagssprache|zum Thema).*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Klicken Sie hier.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Dann finden Sie in dieser Tabelle weitere Informationen.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Achtung:\s*Dieser Link führt.*?(\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Die Informationen sind dann nicht mehr in (Einfacher|Leichter) Sprache.*?(\.|$)', '', text, flags=re.IGNORECASE)

    # 3. Clean up residual double spaces and empty punctuation
    text = re.sub(r'\s+([?.!,;])', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_mediopunkt(text):
    """
    Removes Mediopunkte used for syllable separation.
    """
    return text.replace('·', '').replace('∙', '')

def post_clean_corpus():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    
    total_input = 0
    total_output = 0
    
    for filename in tqdm(files, desc="Post-cleaning corpus"):
        source = filename.replace("_articles.json", "")
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        pairs = data.get("pairs", [])
        total_input += len(pairs)
        
        seen_pairs = set()
        cleaned_pairs = []
        
        for pair in pairs:
            # Clean LS text
            ls_text = pair.get("ls_text", "")
            as_text = pair.get("as_text", "")
            
            # Universal cleaning
            ls_text = normalize_mediopunkt(ls_text)
            if as_text:
                as_text = normalize_mediopunkt(as_text)
            
            # Source specific cleaning for LS
            if source == "brandeins":
                ls_text = clean_brandeins(ls_text)
            elif source == "mdr":
                ls_text = clean_mdr(ls_text)
            elif source == "taz":
                ls_text = clean_taz(ls_text)
                ls_text = clean_taz_hamburg_credits(ls_text)
            elif source == "hamburg":
                ls_text = clean_taz_hamburg_credits(ls_text)
            elif source == "apotheken":
                ls_text = clean_apotheken(ls_text)
            elif source == "hannover":
                ls_text = clean_hannover(ls_text)
            elif source == "stuttgart":
                ls_text = clean_stuttgart(ls_text)
            elif source == "koeln":
                ls_text = clean_stuttgart_koeln(ls_text)
            
            # Source specific cleaning for AS where applicable
            if source == "hannover" and as_text:
                as_text = clean_hannover(as_text)
            
            # Universal navigation & boilerplate removal
            ls_text = clean_navigation_boilerplate(ls_text)
            if as_text:
                as_text = clean_navigation_boilerplate(as_text)
            
            # Standardize whitespace
            ls_text = re.sub(r'\s+', ' ', ls_text).strip()
            as_text = re.sub(r'\s+', ' ', as_text).strip() if as_text else ""
            
            # Filter empty, insufficient texts (< MIN_WORDS) or extreme length ratios
            if not is_valid_length_ratio(ls_text, as_text, MIN_RATIO, MAX_RATIO, MIN_WORDS):
                continue
                
            # Deduplication: only keep first unique (ls_text, as_text) pair
            dedup_key = (ls_text, as_text)
            if dedup_key in seen_pairs:
                continue
            seen_pairs.add(dedup_key)
            
            pair["ls_text"] = ls_text
            pair["as_text"] = as_text
            
            # Also normalize Mediopunkte in as_texts list if present
            if "as_texts" in pair:
                pair["as_texts"] = [normalize_mediopunkt(t) for t in pair["as_texts"]]
                
            cleaned_pairs.append(pair)
            
        total_output += len(cleaned_pairs)
        
        new_data = {
            "source": source,
            "count": len(cleaned_pairs),
            "pairs": cleaned_pairs
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)

    print(f"\nPost-cleaning complete. Final corpus saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    post_clean_corpus()
