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
import unicodedata

# List of filenames or keywords to identify form/consent/list documents that should be excluded completely
EXCLUDE_FILENAMES = {
    # 1. Questionnaires / Surveys
    "ILS KIWA_Bedarfsumfrage AD002 Prüfer.docx",
    
    # 2. Consent / Release of confidentiality forms
    "ILS07 Einwilligung KS DS001 Prüfer.docx",
    "ILS08 BwH Entbindung Schweigepflicht AD002 prüfen.docx",
    "ILS09 BwH Einverständniserklärung AD001 prüfen.docx",
    
    # 3. Pure bullet point lists / Glossaries / Appendices
    "ILS Anlage Resozialisierung Hausordungung HL AD001.docx",
    
    # 4. Content / Topic Mismatches (differing subjects between AS and LS)
    "ILS_CAU_Geologiemuseum AD002 Prüfer Mail.docx",  # LS is Geology Museum, AS is Inclusion Action Day PM
    "07 ÖPNV ILS_AD001 Prüfer.docx",                  # LS is Public Transit, AS source contains Land-use plan
    
    # 5. Extreme Length Asymmetry & Multi-Chapter Manuals
    "ILS Texte MiPi Tablet-Führerschein.docx",        # LS is 10k-word full manual vs 500-word AS intro
    "ILS IIB_MD-Stendal_Textteil_5 001 geprüft.docx", # LS is 511 words vs 73 words AS (7:1 ratio)
    "16 Beirat Senioren ILS_AD001 Prüfer.docx",       # LS is 209 words vs 1019 words generic AS text
    
    # 6. Non-article / Administrative Formats
    "ILS FRAGEN Podium - Parlamentarischer Abend - AD001.docx", # Panel discussion questions / cue sheet
    "ILS_Impressum ZuMiNET LS.docx",                            # Web Impressum / Legal notice
}
EXCLUDE_FILENAMES_NFC = {unicodedata.normalize('NFC', f) for f in EXCLUDE_FILENAMES}

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

# Punctuation and continuation definitions for smart sentence reconstruction
END_PUNCT = ('.', '!', '?', ':', ';', '…', '“', '\"', '”', '»', '«', ')', '–', '-', ',')
CONTINUATION_ENDINGS = {
    'und', 'oder', 'sowie', 'sowohl', 'weder', 'noch', 'aber', 'denn', 'weil', 'dass', 'daß', 'wenn', 'als', 'ob', 'obwohl', 'damit', 'wie',
    'für', 'in', 'im', 'von', 'vom', 'mit', 'zu', 'zur', 'zum', 'auf', 'aus', 'über', 'unter', 'nach', 'bei', 'beim', 'an', 'am', 'vor', 'durch', 'gegen', 'ohne', 'um', 'seit', 'ab', 'bis', 'zwischen', 'hinter', 'neben',
    'der', 'die', 'das', 'des', 'dem', 'den', 'ein', 'eine', 'einer', 'eines', 'einem', 'einen', 'ihr', 'ihre', 'ihrer', 'ihrem', 'ihren', 'sein', 'seine', 'seiner', 'seinem', 'seinen', 'mein', 'meine', 'dein', 'deine', 'unser', 'unsere', 'euer', 'eure',
    'ist', 'sind', 'war', 'waren', 'wird', 'werden', 'wurde', 'wurden', 'hat', 'haben', 'hatte', 'hatten', 'heißt', 'heißen', 'gibt', 'bedeutet', 'gehört', 'gehören', 'kann', 'können', 'muss', 'müssen', 'soll', 'sollen', 'darf', 'dürfen', 'will', 'wollen',
    'auch', 'nicht', 'sehr', 'zum beispiel', 'z. b.', 'z.b.', 'wie zum beispiel'
}

def smart_reconstruct_lines(text):
    if not text:
        return ""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return ""
        
    result = []
    for i, line in enumerate(lines):
        if i == len(lines) - 1:
            if not line.endswith(END_PUNCT):
                line += '.'
            result.append(line)
            break
            
        next_line = lines[i + 1]
        
        # 1. Line already ends with punctuation
        if line.endswith(END_PUNCT):
            result.append(line)
            continue
            
        # 2. Line continues on next line:
        # a) Next line starts with lowercase letter
        # b) Current line ends with a continuation token (conjunction, preposition, auxiliary verb, etc.)
        words = line.lower().split()
        last_word = words[-1] if words else ''
        last_two = ' '.join(words[-2:]) if len(words) >= 2 else ''
        if (next_line and next_line[0].islower()) or last_word in CONTINUATION_ENDINGS or last_two in CONTINUATION_ENDINGS:
            result.append(line)
            continue
            
        # 3. Independent heading, list item, or sentence missing terminal punctuation -> append '.'
        result.append(line + '.')
        
    joined = ' '.join(result)
    joined = re.sub(r'\s+', ' ', joined).strip()
    joined = re.sub(r'\.\s*\.', '.', joined)
    joined = re.sub(r':\.', ':', joined)
    joined = re.sub(r',\.', '.', joined)
    joined = re.sub(r'\?\.', '?', joined)
    joined = re.sub(r'!\.', '!', joined)
    return joined

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
    
    # Clean trailing signature leftovers like "V."
    text = text.strip()
    text = re.sub(r'\n+V\.$', '', text)
    text = re.sub(r'\n+V\n+V\.$', '', text)
    
    # Smart line joining and punctuation reconstruction
    text = smart_reconstruct_lines(text)
    
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
        ls_file_nfc = unicodedata.normalize('NFC', ls_file)
        
        # Check exclusion criteria
        if ls_file_nfc in EXCLUDE_FILENAMES_NFC or ls_file in EXCLUDE_FILENAMES:
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
