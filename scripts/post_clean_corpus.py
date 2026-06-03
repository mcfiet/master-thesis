import json
import os
import re
from tqdm import tqdm

# Configuration
INPUT_DIR = "results/corpus_cleaned"
OUTPUT_DIR = "results/corpus_final"

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

def clean_stuttgart(text):
    """
    Cleans Stuttgart specific artifacts like repeated titles.
    """
    # Example: "Lebenspartnerschaft - Umwandlung in eine Ehe beantragen Umwandlung in eine Ehe beantragen"
    # This is harder to catch without knowing the exact repetition, but we can try to find identical consecutive phrases.
    # For now, let's keep it simple and target common duplicates if possible.
    return text

def normalize_mediopunkt(text):
    """
    Removes Mediopunkte used for syllable separation.
    """
    return text.replace('·', '').replace('∙', '')

def post_clean_corpus():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    
    for filename in tqdm(files, desc="Post-cleaning corpus"):
        source = filename.replace("_articles.json", "")
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        pairs = data.get("pairs", [])
        cleaned_pairs = []
        
        for pair in pairs:
            # Clean LS text
            ls_text = pair.get("ls_text", "")
            
            # Universal cleaning
            ls_text = normalize_mediopunkt(ls_text)
            
            # Source specific cleaning
            if source == "brandeins":
                ls_text = clean_brandeins(ls_text)
            elif source == "mdr":
                ls_text = clean_mdr(ls_text)
            elif source == "taz":
                ls_text = clean_taz(ls_text)
            
            pair["ls_text"] = ls_text
            
            # Also normalize Mediopunkte in AS text for consistency
            if "as_text" in pair:
                pair["as_text"] = normalize_mediopunkt(pair["as_text"])
            if "as_texts" in pair:
                pair["as_texts"] = [normalize_mediopunkt(t) for t in pair["as_texts"]]
                
            cleaned_pairs.append(pair)
            
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
