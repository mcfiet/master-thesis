import os
import json
import glob
import pandas as pd
import textstat
from tqdm import tqdm

# Ensure German language is used for syllable counting and formulas
textstat.set_lang('de')

def calculate_metrics(text):
    if not text or len(text.strip()) == 0:
        return {
            "flesch_reading_ease": None,
            "wiener_sachtextformel": None,
            "lix": None
        }
    
    # Textstat's German implementation of Flesch Reading Ease uses the Amstad formula
    try:
        fre = textstat.flesch_reading_ease(text)
    except Exception:
        fre = None
        
    try:
        # Variant 1 is commonly used for German
        wstf = textstat.wiener_sachtextformel(text, variant=1)
    except Exception:
        wstf = None
        
    try:
        lix = textstat.lix(text)
    except Exception:
        lix = None
        
    return {
        "flesch_reading_ease": fre,
        "wiener_sachtextformel": wstf,
        "lix": lix
    }

def main():
    input_dir = "data/corpus/4_normalized_clean"
    output_file = "data/analysis/readability_analysis.csv"
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    files = glob.glob(os.path.join(input_dir, "*.json"))
    all_results = []
    
    print(f"Processing {len(files)} files for readability analysis...")
    
    for file_path in tqdm(files):
        source_name = os.path.basename(file_path).replace("_articles.json", "")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        pairs = data.get("pairs", [])
        
        for i, pair in enumerate(pairs):
            ls_text = pair.get("ls_text")
            as_text = pair.get("as_text")
            
            if not ls_text or not as_text:
                continue
                
            ls_metrics = calculate_metrics(ls_text)
            as_metrics = calculate_metrics(as_text)
            
            result = {
                "source": source_name,
                "article_index": i,
                "ls_url": pair.get("ls_url"),
                "as_url": pair.get("as_url"),
                
                "ls_flesch": ls_metrics["flesch_reading_ease"],
                "ls_wiener": ls_metrics["wiener_sachtextformel"],
                "ls_lix": ls_metrics["lix"],
                
                "as_flesch": as_metrics["flesch_reading_ease"],
                "as_wiener": as_metrics["wiener_sachtextformel"],
                "as_lix": as_metrics["lix"]
            }
            all_results.append(result)
            
    if not all_results:
        print("No results collected. Check input files.")
        return
        
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)
    print(f"\nReadability analysis complete. Results saved to {output_file}")
    print(f"Total pairs analyzed: {len(all_results)}")

if __name__ == "__main__":
    main()
