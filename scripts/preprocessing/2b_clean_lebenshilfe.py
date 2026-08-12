import json
import os
import re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', required=True)
parser.add_argument('--output_file', required=True)
args = parser.parse_args()

INPUT_FILE = args.input_file
OUTPUT_FILE = args.output_file
# List of filenames or keywords to identify form/consent/list documents that should be excluded completely
EXCLUDE_FILENAMES = {
    # Questionnaire/Surveys
    "ILS KIWA_Bedarfsumfrage AD002 Prüfer.docx",
    # Consent / Release of confidentiality forms
    "ILS07 Einwilligung KS DS001 Prüfer.docx",
    "ILS08 BwH Entbindung Schweigepflicht AD002 prüfen.docx",
    "ILS09 BwH Einverständniserklärung AD001 prüfen.docx",
    # Pure bullet point lists / Glossaries
    "ILS Anlage Resozialisierung Hausordungung HL AD001.docx",
}

# Regex definitions to clean Lebenshilfe signatures, credits, illustrators and formatting noise
CLEANING_PATTERNS = [
    # 1. Typical translator and checker signatures at the end
    (r'(?i)Der Text in Leichter Sprache ist von.*?(\.|$)', ''),
    (r'(?i)Der Text in Leichter Sprache ist vom.*?(\.|$)', ''),
    (r'(?i)Den Text haben.*?geprüft(\.|$)', ''),
    (r'(?i)Den Text in Leichter Sprache haben.*?geprüft(\.|$)', ''),
    (r'(?i)Der Text wurde von der Prüfgruppe.*?geprüft(\.|$)', ''),
    (r'(?i)Diese Prüfenden haben dabei mitgeholfen.*?(\.|$)', ''),
    (r'(?i)Institut für Leichte Sprache.*?(\.|$)', ''),
    (r'(?i)Lebenshilfe Schleswig-Holstein e\.V\..*?(\.|$)', ''),
    (r'(?i)Kehdenstraße\s+\d+-\d+.*?(\.|$)', ''),
    (r'(?i)24103 Kiel.*?(\.|$)', ''),
    
    # 2. Illustrator signatures
    (r'(?i)Die (gezeichneten\s+)?Bilder im Text sind von.*?(\.|$)', ''),
    (r'(?i)Illustrator Stefan Albers, Atelier Fleetinsel.*?(\.|$)', ''),
    (r'(?i)Lebenshilfe für Menschen mit geistiger Behinderung Bremen e\.V\..*?(\.|$)', ''),
    (r'(?i)Justizministerium S-H.*?(\.|$)', ''),
    
    # 3. Inclusion Europe and other logos
    (r'(?i)© European Easy-to-Read Logo: Inclusion Europe.*?(\.|$)', ''),
    (r'(?i)Mehr Informationen im Internet unter: Internetseite von Inclusion Europe.*?(\.|$)', ''),
    (r'(?i)Mehr Informationen im Internet unter:.*?Inclusion Europe.*?(\.|$)', ''),
]

def clean_text(text):
    if not text:
        return ""
    
    # Strip Table of Contents (TOC) at the start of Tablet manual
    # Looks like lines containing a tab character followed by a page number (e.g., "Handbuch ...\t1")
    text = re.sub(r'(?m)^.*?\t\d+\s*$', '', text)
    
    # Normalize syllable separators (Mediopoints)
    text = text.replace('·', '').replace('∙', '')
    
    # Apply standard cleaning regexes
    for pattern, replacement in CLEANING_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.DOTALL)
        
    # Remove screenshots placeholders or annotations often found in LS documents
    text = re.sub(r'(?i)Screenshot\s+[A-Za-z0-9_\-\s]+', '', text)
    text = re.sub(r'(?i)\?\s*Hier Screenshot einfügen\s*\?', '', text)
    text = re.sub(r'(?i)Screenshot\s+einfügen', '', text)
    
    # Remove table-of-contents line artifacts with trailing tab/spaces and page numbers (e.g. "Handbuch ... 1")
    text = re.sub(r'.*?\t\d+\n', '', text)
    
    # Standardize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse multiple newlines into max 2 newlines (paragraph separation)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Clean trailing signature leftovers like "V."
    text = text.strip()
    text = re.sub(r'\n+V\.$', '', text)
    text = re.sub(r'\n+V\n+V\.$', '', text)
    
    return text.strip()

def clean_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
        
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    cleaned_data = []
    excluded_count = 0
    
    for item in data:
        ls_file = item.get("ls_filename", "")
        
        # Check exclusion criteria
        if ls_file in EXCLUDE_FILENAMES:
            print(f"Excluding form/questionnaire file: {ls_file}")
            excluded_count += 1
            continue
            
        # Clean both target (LS) and source (AS) texts
        cleaned_ls = clean_text(item.get("ls_text", ""))
        cleaned_as = clean_text(item.get("as_text", ""))
        
        # Ensure we still have significant text remaining after cleaning
        if len(cleaned_ls.split()) < 10 or len(cleaned_as.split()) < 10:
            print(f"Excluding {ls_file} due to insufficient text length after cleaning.")
            excluded_count += 1
            continue
            
        item["ls_text"] = cleaned_ls
        item["as_text"] = cleaned_as
        cleaned_data.append(item)
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
    print(f"\nProcessing complete:")
    print(f"Total original articles: {len(data)}")
    print(f"Excluded articles: {excluded_count}")
    print(f"Cleaned articles saved: {len(cleaned_data)}")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_dataset()
