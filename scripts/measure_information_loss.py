import json
import os
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch
from collections import Counter

# Configuration
CORPUS_DIR = "results/corpus"
OUTPUT_CSV = "results/information_loss_analysis.csv"
OUTPUT_JSON = "results/information_loss_details.json"
SPACY_MODEL = "de_core_news_lg"
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

def load_models():
    print(f"Loading SpaCy model: {SPACY_MODEL}")
    nlp = spacy.load(SPACY_MODEL)
    
    print(f"Loading SentenceTransformer model: {SBERT_MODEL}")
    sbert = SentenceTransformer(SBERT_MODEL)
    
    return nlp, sbert

def get_entities(doc):
    """Extract entities and their labels."""
    return set([(ent.text.lower(), ent.label_) for ent in doc.ents])

def calculate_ner_recall(as_doc, ls_doc):
    """Calculate how many entities from AS are preserved in LS."""
    as_ents = get_entities(as_doc)
    ls_ents = get_entities(ls_doc)
    
    if not as_ents:
        return 1.0, 0, 0 # Avoid division by zero if AS has no entities
    
    # We count an entity as "preserved" if the text matches exactly (lowered)
    # A more fuzzy matching could be implemented later
    as_texts = set([e[0] for e in as_ents])
    ls_texts = set([e[0] for e in ls_ents])
    
    preserved = as_texts.intersection(ls_texts)
    recall = len(preserved) / len(as_texts)
    
    return recall, len(as_texts), len(ls_texts)

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
            ls_text = pair.get("ls_text", "")
            
            if not as_text or not ls_text:
                continue
                
            # SpaCy Processing
            as_doc = nlp(as_text)
            ls_doc = nlp(ls_text)
            
            # NER Recall
            ner_recall, as_ent_count, ls_ent_count = calculate_ner_recall(as_doc, ls_doc)
            
            # Semantic Similarity
            embeddings = sbert.encode([as_text, ls_text], convert_to_tensor=True)
            sim = util.cos_sim(embeddings[0], embeddings[1]).item()
            
            # Linguistic Metrics
            as_ling = get_linguistic_features(as_doc)
            ls_ling = get_linguistic_features(ls_doc)
            
            # Token Ratio
            token_ratio = ls_ling['total_tokens'] / as_ling['total_tokens'] if as_ling['total_tokens'] > 0 else 0
            
            res = {
                'source': source,
                'as_url': pair.get('as_url'),
                'token_ratio': token_ratio,
                'as_tokens': as_ling['total_tokens'],
                'ls_tokens': ls_ling['total_tokens'],
                'as_avg_sent_len': as_ling['avg_sent_len'],
                'ls_avg_sent_len': ls_ling['avg_sent_len'],
                'ner_recall': ner_recall,
                'as_ent_count': as_ent_count,
                'ls_ent_count': ls_ent_count,
                'semantic_similarity': sim,
                'as_lexical_density': as_ling['lexical_density'],
                'ls_lexical_density': ls_ling['lexical_density'],
                'as_adj_ratio': as_ling['pos_ratios'].get('ADJ', 0),
                'ls_adj_ratio': ls_ling['pos_ratios'].get('ADJ', 0),
                'as_noun_ratio': as_ling['pos_ratios'].get('NOUN', 0) + as_ling['pos_ratios'].get('PROPN', 0),
                'ls_noun_ratio': ls_ling['pos_ratios'].get('NOUN', 0) + ls_ling['pos_ratios'].get('PROPN', 0),
                'as_verb_ratio': as_ling['pos_ratios'].get('VERB', 0),
                'ls_verb_ratio': ls_ling['pos_ratios'].get('VERB', 0),
                'as_conj_ratio': as_ling['pos_ratios'].get('CONJ', 0) + as_ling['pos_ratios'].get('CCONJ', 0) + as_ling['pos_ratios'].get('SCONJ', 0),
                'ls_conj_ratio': ls_ling['pos_ratios'].get('CONJ', 0) + ls_ling['pos_ratios'].get('CCONJ', 0) + ls_ling['pos_ratios'].get('SCONJ', 0)
            }
            all_results.append(res)
            
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Summary statistics by source
    summary = df.groupby('source').agg({
        'token_ratio': 'mean',
        'ner_recall': 'mean',
        'semantic_similarity': 'mean',
        'as_tokens': 'mean',
        'ls_tokens': 'mean'
    }).reset_index()
    
    print("\n--- Summary Statistics ---")
    print(summary)
    
    # Save detailed results as JSON as well
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    analyze_corpus()
