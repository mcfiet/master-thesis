import os
import json
import glob
import pandas as pd
import spacy
from tqdm import tqdm

# Load German spaCy model
try:
    nlp = spacy.load("de_core_news_sm")
except OSError:
    # Fallback if the model is named differently or not linked correctly
    import de_core_news_sm
    nlp = de_core_news_sm.load()

def calculate_ttr_metrics(text, window_size=50):
    if not text or len(text.strip()) == 0:
        return {
            "tokens": 0,
            "ttr": None,
            "mattr": None
        }
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Filter: Only keep alphanumeric tokens, remove punctuation
    # We use lemmatized forms for TTR calculation
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    
    token_count = len(tokens)
    if token_count == 0:
        return {"tokens": 0, "ttr": None, "mattr": None}
    
    # 1. Classical TTR
    unique_types = len(set(tokens))
    ttr = unique_types / token_count
    
    # 2. MATTR (Moving Average TTR)
    if token_count < window_size:
        mattr = ttr  # Fallback to TTR for short texts
    else:
        ttr_values = []
        for i in range(token_count - window_size + 1):
            window = tokens[i : i + window_size]
            window_ttr = len(set(window)) / window_size
            ttr_values.append(window_ttr)
        mattr = sum(ttr_values) / len(ttr_values)
    
    return {
        "tokens": token_count,
        "ttr": ttr,
        "mattr": mattr
    }

def process_corpus(input_dir, output_file):
    files = glob.glob(os.path.join(input_dir, "*.json"))
    results = []
    
    print(f"Processing {len(files)} corpus files...")
    
    for file_path in tqdm(files):
        source_name = os.path.basename(file_path).replace("_articles.json", "")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            pairs = data.get("pairs", [])
            
        for article in pairs:
            as_text = article.get("as_text")
            if as_text is None:
                as_texts = article.get("as_texts", [])
                as_text = "\n".join(as_texts) if as_texts else ""
                
            ls_text = article.get("ls_text")
            if ls_text is None:
                ls_texts = article.get("ls_texts", [])
                ls_text = "\n".join(ls_texts) if ls_texts else ""
                
            url = article.get("url") or article.get("ls_url") or "unknown"
            
            as_metrics = calculate_ttr_metrics(as_text)
            ls_metrics = calculate_ttr_metrics(ls_text)
            
            results.append({
                "source": source_name,
                "url": url,
                "as_tokens": as_metrics["tokens"],
                "as_ttr": as_metrics["ttr"],
                "as_mattr": as_metrics["mattr"],
                "ls_tokens": ls_metrics["tokens"],
                "ls_ttr": ls_metrics["ttr"],
                "ls_mattr": ls_metrics["mattr"]
            })
            
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Analysis saved to {output_file}")

if __name__ == "__main__":
    INPUT_DIR = "data/corpus/4_normalized_clean"
    OUTPUT_FILE = "data/analysis/ttr_analysis.csv"
    process_corpus(INPUT_DIR, OUTPUT_FILE)
