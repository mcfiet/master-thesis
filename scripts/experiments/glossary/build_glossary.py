import os
import glob
import json
import urllib.request
import urllib.parse
import time
from collections import Counter

# Arbeitsverzeichnis wird beibehalten, Pfade werden normal relativ angegeben

# Define fallback glossary for words that are not on Hurraki or fail to fetch
FALLBACK_GLOSSARY = {
    "landeshauptstadt": "Die wichtigste Stadt in einem Bundesland. Dort arbeitet die Regierung.",
    "schleswig-holstein": "Ein Bundesland im Norden von Deutschland.",
    "niedersachsen": "Ein Bundesland in Deutschland.",
    "öffnungszeiten": "Die Zeiten, an denen ein Amt oder Geschäft geöffnet hat.",
    "ansprechpartner": "Eine Person, die man bei Fragen anrufen oder anschreiben kann.",
    "dienstleistungen": "Angebote oder Hilfen von einem Amt oder einer Firma.",
    "pressemitteilung": "Ein Text mit Informationen für die Zeitung.",
    "beratungsstelle": "Ein Ort, an dem man Hilfe und Informationen von Experten bekommt.",
    "schwerbehinderung": "Eine besonders starke Behinderung, die das Leben schwerer macht.",
    "antragstellung": "Das Ausfüllen und Abgeben von Formularen, um Hilfe zu bekommen.",
    "geschäftsstelle": "Das Büro oder die Zentrale von einem Verein oder einer Firma.",
    "selbstbestimmung": "Selbst entscheiden zu können, wie man leben möchte.",
    "barrierefreiheit": "Wenn es keine Hindernisse gibt. Zum Beispiel Rampen für Rollstühle.",
    "mitarbeiterinnen": "Frauen, die in einer Firma oder einem Amt arbeiten.",
    "eingliederungshilfe": "Geld und Unterstützung vom Staat für Menschen mit Behinderung.",
    "schwerbehindertenausweis": "Ein Ausweis, der zeigt, dass man eine starke Behinderung hat.",
    "antrag": "Ein Formular, mit dem man Geld oder Hilfe fordert.",
    "begleitperson": "Eine Person, die mitgeht, um zu helfen.",
    "bundesregierung": "Die Politiker, die Deutschland regieren.",
    "behinderungen": "Körperliche oder geistige Einschränkungen."
}

def extract_top_words(corpus_dir="data/corpus/4_normalized_clean", num_words=100):
    print("Analysiere Korpus im Ordner:", corpus_dir)
    json_files = glob.glob(os.path.join(corpus_dir, "*.json"))
    if os.path.exists("data/analysis/corpus_master.json"):
        json_files.append("data/analysis/corpus_master.json")
    word_counts = Counter()
    
    # German stop words to filter out
    stop_words = {
        "dass", "weil", "wenn", "aber", "oder", "und", "der", "die", "das", "ein", "eine", 
        "einer", "eines", "einem", "einen", "dem", "den", "des", "mit", "von", "nach", "fuer",
        "über", "unter", "durch", "ohne", "gegen", "wieder", "immer", "schon", "noch", "mehr",
        "oder", "sich", "nicht", "auch", "sind", "wurde", "wurden", "haben", "hatte", "hatten",
        "dies", "diese", "dieser", "diesem", "diesen", "dieses", "kann", "können", "müssen",
        "sollen", "wollen", "dürfen", "werde", "werden", "wird", "können", "gibt", "geben"
    }

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data if isinstance(data, list) else data.get("pairs", [])
            for item in items:
                as_text = item.get("as_text", "").strip()
                if as_text:
                    # Clean and tokenize
                    words = as_text.lower().replace(",", "").replace(".", "").replace("?", "").replace("!", "").split()
                    for word in words:
                        # Clean word
                        word = word.strip("()\"'„“»«-")
                        if word.isalpha() and len(word) >= 9 and word not in stop_words:
                            word_counts[word] += 1
                            
    # Return top N words
    return [word for word, count in word_counts.most_common(num_words)]

def fetch_hurraki_definition(word):
    url = f"https://hurraki.de/w/api.php?action=query&format=json&prop=extracts&exintro=1&explaintext=1&titles={urllib.parse.quote(word.capitalize())}"
    
    # Use browser headers to prevent 403 Forbidden
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id != "-1":
                    extract = page_data.get("extract", "").strip()
                    if extract:
                        # Clean up extract text (replace multiple newlines with spaces)
                        extract_clean = " ".join([line.strip() for line in extract.splitlines() if line.strip()])
                        return extract_clean
    except Exception as e:
        print(f"Error fetching '{word}': {e}")
    return None

def main():
    top_words = extract_top_words()
    print(f"Top 100 komplexe Wörter extrahiert: {top_words[:10]}...")
    
    glossary = {}
    
    print("\nStarte Abfrage von Hurraki API...")
    for idx, word in enumerate(top_words):
        print(f"[{idx+1}/100] Frage '{word}' ab...")
        definition = fetch_hurraki_definition(word)
        
        if definition:
            print(f"  -> Gefunden auf Hurraki!")
            glossary[word] = definition
        elif word in FALLBACK_GLOSSARY:
            print(f"  -> Nutze Fallback-Erklärung.")
            glossary[word] = FALLBACK_GLOSSARY[word]
        else:
            print(f"  -> Keine Erklärung gefunden.")
            
        # Polite delay to prevent rate limits
        time.sleep(0.5)
        
    # Save glossary
    output_path = "data/vocabs/hurraki_glossary.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(glossary, f, indent=2, ensure_ascii=False)
        
    print(f"\nErfolgreich Glossar mit {len(glossary)} Begriffen gespeichert unter: {output_path}")

if __name__ == "__main__":
    main()
