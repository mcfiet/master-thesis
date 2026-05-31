import json
import os
import glob
import re
import tiktoken
from collections import Counter

# Initialize tiktoken encoding (GPT-4 / GPT-3.5 turbo)
encoding = tiktoken.get_encoding("cl100k_base")

def tokenize_words(text):
    return re.findall(r'[\w∙-]+', text.lower())

def count_llm_tokens(text):
    if not text:
        return 0
    return len(encoding.encode(text))

def count_sentences(text):
    if not text:
        return 0
    sentences = re.split(r'[.!?](?:\s+|$)|(?:\n\s*\n)', text)
    return len([s for s in sentences if s.strip()])

def analyze_corpus(corpus_dir="results/corpus", output_path="research/corpus_statistics.md"):
    files = glob.glob(os.path.join(corpus_dir, "*_articles.json"))
    
    overall_stats = {
        'ls': {'articles': 0, 'tokens': 0, 'words': 0, 'sentences': 0, 'chars': 0, 'vocab': Counter()},
        'as': {'articles': 0, 'tokens': 0, 'words': 0, 'sentences': 0, 'chars': 0, 'vocab': Counter()}
    }
    
    sources_stats = {}

    for file_path in sorted(files):
        source_name = os.path.basename(file_path).replace("_articles.json", "")
        
        source_data = {
            'ls': {'articles': 0, 'tokens': 0, 'words': 0, 'sentences': 0, 'chars': 0, 'vocab': Counter()},
            'as': {'articles': 0, 'tokens': 0, 'words': 0, 'sentences': 0, 'chars': 0, 'vocab': Counter()}
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                pairs = data.get('pairs', []) if isinstance(data, dict) else data
                
                for pair in pairs:
                    for lang in ['ls', 'as']:
                        text = pair.get(f'{lang}_text', '')
                        if not text:
                            texts = pair.get(f'{lang}_texts', [])
                            if isinstance(texts, list):
                                text = '\n'.join(texts)
                            elif isinstance(texts, str):
                                text = texts
                                
                        if not text:
                            continue
                            
                        words = tokenize_words(text)
                        tokens_count = count_llm_tokens(text)
                        sentences = count_sentences(text)
                        
                        source_data[lang]['articles'] += 1
                        source_data[lang]['words'] += len(words)
                        source_data[lang]['tokens'] += tokens_count
                        source_data[lang]['sentences'] += sentences
                        source_data[lang]['chars'] += len(text)
                        source_data[lang]['vocab'].update(words)
                        
                        overall_stats[lang]['articles'] += 1
                        overall_stats[lang]['words'] += len(words)
                        overall_stats[lang]['tokens'] += tokens_count
                        overall_stats[lang]['sentences'] += sentences
                        overall_stats[lang]['chars'] += len(text)
                        overall_stats[lang]['vocab'].update(words)
                        
            sources_stats[source_name] = source_data
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Generate Markdown Table
    md_content = [
        "# Corpus Statistics Overview",
        "",
        "Diese Datei wird automatisch durch das Skript `scripts/corpus_stats.py` generiert. Sie enthält die zusammenfassenden Statistiken für alle Quellen im Korpus, aufgeschlüsselt nach Leichter Sprache (LS) und Alltagssprache (AS).",
        "Die Spalte 'Tokens' verwendet nun `tiktoken` (cl100k_base), was der tatsächlichen Token-Anzahl für LLMs (z.B. GPT-4) entspricht.",
        "",
        "| Source | Pairs | Words (LS) | Words (AS) | Tokens (LS) | Tokens (AS) | Sentences (LS) | Sentences (AS) | Vocab (LS) | Vocab (AS) | TTR (LS) | TTR (AS) | W/S (LS) | W/S (AS) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    ]
    
    for source, stats in sorted(sources_stats.items()):
        ls = stats['ls']
        as_ = stats['as']
        
        pairs = ls['articles']
        
        ls_ws = f"{ls['words'] / ls['sentences']:.1f}" if ls['sentences'] > 0 else "0"
        as_ws = f"{as_['words'] / as_['sentences']:.1f}" if as_['sentences'] > 0 else "0"
        
        ls_ttr = f"{len(ls['vocab']) / ls['words']:.3f}" if ls['words'] > 0 else "0"
        as_ttr = f"{len(as_['vocab']) / as_['words']:.3f}" if as_['words'] > 0 else "0"
        
        md_content.append(f"| {source} | {pairs} | {ls['words']} | {as_['words']} | {ls['tokens']} | {as_['tokens']} | {ls['sentences']} | {as_['sentences']} | {len(ls['vocab'])} | {len(as_['vocab'])} | {ls_ttr} | {as_ttr} | {ls_ws} | {as_ws} |")
    
    # Overall row
    ls_o = overall_stats['ls']
    as_o = overall_stats['as']
    pairs_o = ls_o['articles']
    
    ls_ws_o = f"{ls_o['words'] / ls_o['sentences']:.1f}" if ls_o['sentences'] > 0 else "0"
    as_ws_o = f"{as_o['words'] / as_o['sentences']:.1f}" if as_o['sentences'] > 0 else "0"
    
    ls_ttr_o = f"{len(ls_o['vocab']) / ls_o['words']:.3f}" if ls_o['words'] > 0 else "0"
    as_ttr_o = f"{len(as_o['vocab']) / as_o['words']:.3f}" if as_o['words'] > 0 else "0"
    
    md_content.append(f"| **TOTAL** | **{pairs_o}** | **{ls_o['words']}** | **{as_o['words']}** | **{ls_o['tokens']}** | **{as_o['tokens']}** | **{ls_o['sentences']}** | **{as_o['sentences']}** | **{len(ls_o['vocab'])}** | **{len(as_o['vocab'])}** | **{ls_ttr_o}** | **{as_ttr_o}** | **{ls_ws_o}** | **{as_ws_o}** |")

    # Write to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content) + '\n')
        print(f"Statistics successfully written to {output_path}")
    except Exception as e:
        print(f"Error writing to {output_path}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="results/corpus")
    parser.add_argument('--output_file', default="research/corpus_statistics.md")
    args = parser.parse_args()
    
    analyze_corpus(corpus_dir=args.input_dir, output_path=args.output_file)
