import json
import os
import glob
import re

def linguistic_tokenize(text):
    """
    Standard linguistic tokenization:
    - Splits words from punctuation
    - Handles numbers and common abbreviations
    """
    if not text:
        return []
    # This regex identifies words (including umlauts) and individual punctuation marks
    token_pattern = r'\w+|[^\w\s]'
    return re.findall(token_pattern, text)

def count_corpus_tokens():
    corpus_path = "data/corpus/2_raw_scraped/*.json"
    files = glob.glob(corpus_path)
    
    if not files:
        print(f"No files found in {corpus_path}")
        return

    total_ls_tokens = 0
    total_as_tokens = 0

    print(f"{'Source':<30} | {'LS Tokens':>12} | {'AS Tokens':>12} | {'Total':>12}")
    print("-" * 75)

    for file_path in sorted(files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        file_ls_tokens = 0
        file_as_tokens = 0
        
        pairs = data.get('pairs', [])
        for pair in pairs:
            # Handle LS text
            ls_text = pair.get('ls_text', '')
            if ls_text:
                file_ls_tokens += len(linguistic_tokenize(ls_text))
            
            # Handle AS text (can be a string or a list of strings like in TAZ)
            if 'as_texts' in pair:
                # TAZ structure: array of texts
                as_texts = pair.get('as_texts', [])
                for text in as_texts:
                    if text:
                        file_as_tokens += len(linguistic_tokenize(text))
            elif 'as_text' in pair:
                # Standard structure: single string
                as_text = pair.get('as_text', '')
                if as_text:
                    file_as_tokens += len(linguistic_tokenize(as_text))
        
        source_name = os.path.basename(file_path).replace("_articles.json", "")
        print(f"{source_name:<30} | {file_ls_tokens:>12,} | {file_as_tokens:>12,} | {file_ls_tokens + file_as_tokens:>12,}")
        
        total_ls_tokens += file_ls_tokens
        total_as_tokens += file_as_tokens

    print("-" * 75)
    print(f"{'TOTAL':<30} | {total_ls_tokens:>12,} | {total_as_tokens:>12,} | {total_ls_tokens + total_as_tokens:>12,}")
    print("\nNote: These are 'linguistic tokens' (words and punctuation counted separately).")

if __name__ == "__main__":
    count_corpus_tokens()
