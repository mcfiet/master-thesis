import json
import os
import pandas as pd
import spacy


from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch
from collections import Counter

# Configuration
CORPUS_DIR = "data/corpus/2_raw_scraped"
OUTPUT_CSV = "data/analysis/information_loss_analysis.csv"
OUTPUT_JSON = "data/analysis/information_loss_details.json"
OUTPUT_EXTREMES = "data/analysis/similarity_extremes.json"
SPACY_MODEL = "de_core_news_lg"
SBERT_MODEL = "jinaai/jina-embeddings-v2-base-de"

def load_models():
    print(f"Loading SpaCy model: {SPACY_MODEL}")
    nlp = spacy.load(SPACY_MODEL)
    # Ensure very long articles are not skipped
    nlp.max_length = 2000000
    
    print(f"Loading SentenceTransformer model: {SBERT_MODEL}")
    sbert = SentenceTransformer(SBERT_MODEL, trust_remote_code=True)
    
    # Use Jina's full context window (8192 tokens)
    sbert.max_seq_length = 8192 
    print(f"SBERT max_seq_length set to: {sbert.max_seq_length}")
    
    return nlp, sbert

def get_entities(doc):
    """Extract entities and their labels."""
    return set([(ent.text.lower(), ent.label_) for ent in doc.ents])

def calculate_ner_recall(as_doc, ls_doc):
    """Calculate entity recall in both directions."""
    as_ents = get_entities(as_doc)
    ls_ents = get_entities(ls_doc)
    
    as_texts = set([e[0] for e in as_ents])
    ls_texts = set([e[0] for e in ls_ents])
    
    # AS -> LS (Standard Recall)
    recall_as_ls = len(as_texts.intersection(ls_texts)) / len(as_texts) if as_texts else 1.0
    
    # LS -> AS (Inverted Recall: Does LS invent facts?)
    recall_ls_as = len(ls_texts.intersection(as_texts)) / len(ls_texts) if ls_texts else 1.0
    
    return recall_as_ls, recall_ls_as, len(as_texts), len(ls_texts)

def get_linguistic_features(doc):
    """Calculate POS distribution, lexical density, and average sentence length."""
    pos_counts = Counter([token.pos_ for token in doc])
    total_tokens = len(doc)
    num_sents = len(list(doc.sents))
    
    avg_sent_len = total_tokens / num_sents if num_sents > 0 else 0
    
    # Lexical density: (Nouns + Verbs + Adjectives + Adverbs) / Total Tokens
    content_pos = {'NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'}
    content_words = sum(pos_counts[pos] for pos in content_pos)
    lexical_density = content_words / total_tokens if total_tokens > 0 else 0
    
    return {
        'total_tokens': total_tokens,
        'lexical_density': lexical_density,
        'avg_sent_len': avg_sent_len,
        'pos_ratios': {pos: count / total_tokens for pos, count in pos_counts.items()}
    }

def analyze_corpus():
    nlp, sbert = load_models()
    
    all_results = []
    
    corpus_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".json")]
    
    for filename in corpus_files:
        source = filename.replace("_articles.json", "")
        filepath = os.path.join(CORPUS_DIR, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        pairs = data.get("pairs", [])
        print(f"Processing {len(pairs)} pairs from {source}...")
        
        # Limit for testing/speed if needed
        # pairs = pairs[:10] 
        
        for pair in tqdm(pairs):
            as_text = pair.get("as_text", "")
            # Support plural format (1-to-N alignment) for TAZ etc.
            if not as_text and "as_texts" in pair:
                as_text = "\n\n".join(pair["as_texts"])
                
            ls_text = pair.get("ls_text", "")
            
            if not as_text or not ls_text:
                continue
                
            # SpaCy Processing
            as_doc = nlp(as_text)
            ls_doc = nlp(ls_text)
            
            # NER Recall (Bidirectional)
            ner_recall_as_ls, ner_recall_ls_as, as_ent_count, ls_ent_count = calculate_ner_recall(as_doc, ls_doc)
            
            # Semantic Similarity (Jina Model at different sequence lengths)
            
            # 1. Full Context (8192)
            sbert.max_seq_length = 8192
            embeddings_full = sbert.encode([as_text, ls_text], batch_size=1, convert_to_tensor=True)
            sim_8192 = util.cos_sim(embeddings_full[0], embeddings_full[1]).item()
            
            # 2. Context 512
            sbert.max_seq_length = 512
            embeddings_512 = sbert.encode([as_text, ls_text], batch_size=1, convert_to_tensor=True)
            sim_512 = util.cos_sim(embeddings_512[0], embeddings_512[1]).item()
            
            # 3. Context 128
            sbert.max_seq_length = 128
            embeddings_128 = sbert.encode([as_text, ls_text], batch_size=1, convert_to_tensor=True)
            sim_128 = util.cos_sim(embeddings_128[0], embeddings_128[1]).item()
            
            # Restore max_seq_length for next iteration just in case
            sbert.max_seq_length = 8192
            
            # Linguistic Metrics
            as_ling = get_linguistic_features(as_doc)
            ls_ling = get_linguistic_features(ls_doc)
            
            # Token Ratio
            token_ratio = ls_ling['total_tokens'] / as_ling['total_tokens'] if as_ling['total_tokens'] > 0 else 0
            
            res = {
                'source': source,
                'as_url': pair.get('as_url'),
                'ls_url': pair.get('ls_url'),
                'token_ratio': token_ratio,
                'as_tokens': as_ling['total_tokens'],
                'ls_tokens': ls_ling['total_tokens'],
                'as_avg_sent_len': as_ling['avg_sent_len'],
                'ls_avg_sent_len': ls_ling['avg_sent_len'],
                'ner_recall_as_ls': ner_recall_as_ls,
                'ner_recall_ls_as': ner_recall_ls_as,
                'as_ent_count': as_ent_count,
                'ls_ent_count': ls_ent_count,
                'semantic_similarity_8192': sim_8192,
                'semantic_similarity_512': sim_512,
                'semantic_similarity_128': sim_128,
                'as_lexical_density': as_ling['lexical_density'],
                'ls_lexical_density': ls_ling['lexical_density'],
                'as_adj_ratio': as_ling['pos_ratios'].get('ADJ', 0),
                'ls_adj_ratio': ls_ling['pos_ratios'].get('ADJ', 0),
                'as_noun_ratio': as_ling['pos_ratios'].get('NOUN', 0) + as_ling['pos_ratios'].get('PROPN', 0),
                'ls_noun_ratio': ls_ling['pos_ratios'].get('NOUN', 0) + ls_ling['pos_ratios'].get('PROPN', 0),
                'as_verb_ratio': as_ling['pos_ratios'].get('VERB', 0),
                'ls_verb_ratio': ls_ling['pos_ratios'].get('VERB', 0),
                'as_conj_ratio': as_ling['pos_ratios'].get('CONJ', 0) + as_ling['pos_ratios'].get('CCONJ', 0) + as_ling['pos_ratios'].get('SCONJ', 0),
                'ls_conj_ratio': ls_ling['pos_ratios'].get('CONJ', 0) + ls_ling['pos_ratios'].get('CCONJ', 0) + ls_ling['pos_ratios'].get('SCONJ', 0),
                'as_text': as_text[:1000], # Store preview for audit
                'ls_text': ls_text[:1000]
            }
            all_results.append(res)
            
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Correlation Analysis (using 512 for now since 8192 is skipped)
    correlation = df['token_ratio'].corr(df['semantic_similarity_512'])
    print(f"\nCorrelation between Token Ratio and Semantic Similarity (512): {correlation:.4f}")
    
    # Extremes Extraction
    print("Extracting extremes for manual audit...")
    sorted_df = df.sort_values(by='semantic_similarity_512')
    bottom_5 = sorted_df.head(5)
    top_5 = sorted_df.tail(5)
    extremes = pd.concat([bottom_5, top_5])
    
    extremes.to_json(OUTPUT_EXTREMES, orient='records', indent=4, force_ascii=False)
    print(f"Extremes saved to {OUTPUT_EXTREMES}")

    # Summary statistics by source
    summary = df.groupby('source').agg({
        'token_ratio': 'mean',
        'ner_recall_as_ls': 'mean',
        'ner_recall_ls_as': 'mean',
        'semantic_similarity_128': 'mean',
        'semantic_similarity_512': 'mean',
        'as_tokens': 'mean',
        'ls_tokens': 'mean'
    }).reset_index()
    
    print("\n--- Summary Statistics ---")
    print(summary)
    
    # Save detailed results as JSON as well
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="data/corpus/raw")
    parser.add_argument('--output_csv', default="data/analysis/information_loss_analysis.csv")
    args = parser.parse_args()
    
    CORPUS_DIR = args.input_dir
    OUTPUT_CSV = args.output_csv
    
    analyze_corpus()
