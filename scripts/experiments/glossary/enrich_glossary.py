import os
import glob
import json
import re

# Arbeitsverzeichnis wird beibehalten, Pfade werden normal relativ angegeben

def load_glossary(path="data/vocabs/hurraki_glossary.json"):
    if not os.path.exists(path):
        print(f"Glossar nicht gefunden unter {path}. Bitte zuerst das build_hurraki_glossary.py Skript ausführen.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_word(word):
    return word.lower().strip("()\"'„“»«-.,?!;:")

def enrich_corpus(glossary, input_dir="data/corpus/4_normalized_clean", output_dir="data/corpus/5_glossary_enriched"):
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    total_original = 0
    total_added = 0
    
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        source = data.get("source", "unknown")
        original_pairs = data.get("pairs", [])
        enriched_pairs = []
        
        for pair in original_pairs:
            as_text = pair.get("as_text", "").strip()
            ls_text = pair.get("ls_text", "").strip()
            
            if not as_text or not ls_text:
                continue
                
            # Always keep the original pair
            enriched_pairs.append(pair)
            total_original += 1
            
            # Find which glossary words are present in this Alltagssprache text
            words_in_text = set()
            # Split text into clean words
            for w in re.findall(r'\b\w+\b', as_text):
                cleaned = clean_word(w)
                if cleaned in glossary:
                    words_in_text.add(cleaned)
                    
            if words_in_text:
                # Build explanations string to append
                explanations = []
                for word in sorted(words_in_text):
                    # Make word first letter capitalized
                    cap_word = word.capitalize()
                    expl = glossary[word]
                    explanations.append(f"{cap_word}. {cap_word} bedeutet: {expl}")
                
                # Create a new augmented pair where the target has the explanations appended at the end
                explanation_suffix = "\n\n" + "\n".join(explanations)
                
                augmented_pair = {
                    "as_text": as_text,
                    "ls_text": ls_text + explanation_suffix,
                    "augmented": True
                }
                
                # Check for other keys like article_id if present
                for key in pair:
                    if key not in ["as_text", "ls_text"]:
                        augmented_pair[key] = pair[key]
                        
                enriched_pairs.append(augmented_pair)
                total_added += 1
                
        # Save enriched pairs to the output directory
        new_data = {
            "source": source,
            "pairs": enriched_pairs
        }
        
        output_file_path = os.path.join(output_dir, os.path.basename(file_path))
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
            
    print(f"Augmentierung abgeschlossen!")
    print(f"Originale Paare behalten: {total_original}")
    print(f"Augmentierte Paare mit Erklärungen hinzugefügt: {total_added}")
    print(f"Gesamtanzahl Paare im angereicherten Korpus: {total_original + total_added}")

def main():
    glossary = load_glossary()
    if not glossary:
        return
        
    print(f"Glossar mit {len(glossary)} Begriffen geladen. Starte Korpus-Augmentierung...")
    enrich_corpus(glossary)

if __name__ == "__main__":
    main()
