import json
import os
import glob

def summarize():
    corpus_dir = "data/corpus/raw"
    files = glob.glob(os.path.join(corpus_dir, "*_articles.json"))
    
    total_pairs = 0
    total_ls_tokens = 0
    total_as_tokens = 0
    
    print(f"{'Source':<30} | {'Pairs':<6} | {'LS Tokens':<10} | {'AS Tokens':<10} | {'Ratio (LS/AS)':<10}")
    print("-" * 75)
    
    for file_path in sorted(files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Check format (some might be a list directly, some a dict with 'pairs')
                pairs = data.get('pairs', []) if isinstance(data, dict) else data
                
                if not pairs:
                    continue
                    
                source_name = os.path.basename(file_path).replace("_articles.json", "")
                
                pairs_count = len(pairs)
                ls_tokens = sum(p.get('ls_tokens', 0) for p in pairs)
                as_tokens = sum(p.get('as_tokens', 0) for p in pairs)
                
                ratio = round(ls_tokens / as_tokens, 2) if as_tokens > 0 else 0
                
                print(f"{source_name:<30} | {pairs_count:<6} | {ls_tokens:<10} | {as_tokens:<10} | {ratio:<10}")
                
                total_pairs += pairs_count
                total_ls_tokens += ls_tokens
                total_as_tokens += as_tokens
        except Exception as e:
            pass

    print("-" * 75)
    total_ratio = round(total_ls_tokens / total_as_tokens, 2) if total_as_tokens > 0 else 0
    print(f"{'TOTAL':<30} | {total_pairs:<6} | {total_ls_tokens:<10} | {total_as_tokens:<10} | {total_ratio:<10}")

if __name__ == "__main__":
    summarize()
