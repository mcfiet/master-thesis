#!/usr/bin/env python3
"""
Builds a unified Master CSV file from a directory of matched text pairs (JSON).
Combines:
- Basic metadata (source, URLs, raw text)
- Semantic similarity (SBERT Jina 8192)
- NER Recall (AS -> LS, LS -> AS)
- Readability metrics (Flesch Reading Ease, Wiener Sachtextformel, LIX)
- Lexical diversity (TTR, MATTR)
"""

import os
import json
import glob
import re
import argparse
import sys
import types

# Hack to prevent ModuleNotFoundError: No module named 'transformers.onnx' in newer transformers versions
try:
    import transformers.onnx
except ModuleNotFoundError:
    import sys
    mock_onnx = types.ModuleType("transformers.onnx")
    mock_onnx.OnnxConfig = object
    sys.modules["transformers.onnx"] = mock_onnx

import pandas as pd
import spacy
import textstat
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch
from collections import Counter

# Ensure German language for readability formulas
textstat.set_lang('de')

def parse_args():
    parser = argparse.ArgumentParser(description="Build a comprehensive master CSV of a corpus.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/corpus/4_normalized_clean",
        help="Path to directory containing source JSON files."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/analysis/corpus_master.csv",
        help="Path to save the generated Master CSV."
    )
    parser.add_argument(
        "--sbert_model",
        type=str,
        default="jinaai/jina-embeddings-v2-base-de",
        help="SBERT model to use for semantic similarity."
    )
    parser.add_argument(
        "--spacy_model",
        type=str,
        default="de_core_news_lg",
        help="SpaCy model for linguistic parsing and NER."
    )
    return parser.parse_args()

def get_entities(doc):
    return set([(ent.text.lower(), ent.label_) for ent in doc.ents])

def calculate_ner_recall(as_doc, ls_doc):
    as_ents = get_entities(as_doc)
    ls_ents = get_entities(ls_doc)
    as_texts = set([e[0] for e in as_ents])
    ls_texts = set([e[0] for e in ls_ents])
    
    recall_as_ls = len(as_texts.intersection(ls_texts)) / len(as_texts) if as_texts else 1.0
    recall_ls_as = len(ls_texts.intersection(as_texts)) / len(ls_texts) if ls_texts else 1.0
    return recall_as_ls, recall_ls_as, len(as_texts), len(ls_texts)

def calculate_readability(text):
    if not text or len(text.strip()) == 0:
        return None, None, None
    try:
        fre = textstat.flesch_reading_ease(text)
    except Exception:
        fre = None
    try:
        wstf = textstat.wiener_sachtextformel(text, variant=1)
    except Exception:
        wstf = None
    try:
        lix = textstat.lix(text)
    except Exception:
        lix = None
    return fre, wstf, lix

def calculate_ttr_metrics(doc, window_size=50):
    # Only keep alphanumeric tokens, remove punctuation, use lemmatized forms
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    token_count = len(tokens)
    if token_count == 0:
        return 0, None, None
    
    # 1. Classical TTR
    unique_types = len(set(tokens))
    ttr = unique_types / token_count
    
    # 2. MATTR (Moving Average TTR)
    if token_count < window_size:
        mattr = ttr
    else:
        ttr_values = []
        for i in range(token_count - window_size + 1):
            window = tokens[i : i + window_size]
            window_ttr = len(set(window)) / window_size
            ttr_values.append(window_ttr)
        mattr = sum(ttr_values) / len(ttr_values)
        
    return token_count, ttr, mattr

def main():
    args = parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    # 1. Load Models
    print(f"Loading SpaCy model: {args.spacy_model}...")
    try:
        nlp = spacy.load(args.spacy_model)
    except OSError:
        print(f"Warning: Could not load '{args.spacy_model}'. Trying fallback models...")
        fallback_models = ["de_core_news_md", "de_core_news_sm"]
        nlp = None
        for model in fallback_models:
            try:
                print(f"Trying to load {model}...")
                nlp = spacy.load(model)
                print(f"Successfully loaded {model} as fallback.")
                break
            except OSError:
                continue
        if nlp is None:
            print(f"\nError: Could not load any German SpaCy model ({args.spacy_model} or fallbacks).")
            print("Please download a model using: python -m spacy download de_core_news_lg (oder de_core_news_sm)")
            return
    nlp.max_length = 2000000
    
    print(f"Loading SentenceTransformer: {args.sbert_model}...")
    sbert = SentenceTransformer(args.sbert_model, trust_remote_code=True)
    sbert.max_seq_length = 8192
    
    # 2. Find JSON files
    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in '{args.input_dir}'")
        return
        
    print(f"Processing {len(json_files)} source files from '{args.input_dir}'...")
    all_rows = []
    
    for file_path in json_files:
        source_name = os.path.basename(file_path).replace("_articles.json", "")
        print(f"Analyzing source: {source_name}...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            pairs = data.get("pairs", []) if isinstance(data, dict) else data
            
        for pair in tqdm(pairs, desc=source_name):
            as_text = pair.get("as_text", "").strip()
            ls_text = pair.get("ls_text", "").strip()
            
            if not as_text or not ls_text:
                continue
                
            as_url = pair.get("as_url") or pair.get("url") or "unknown"
            ls_url = pair.get("ls_url") or "unknown"
            
            # Compute NLP Docs
            as_doc = nlp(as_text)
            ls_doc = nlp(ls_text)
            
            # 1. Similarity
            emb_as = sbert.encode(as_text, convert_to_tensor=True)
            emb_ls = sbert.encode(ls_text, convert_to_tensor=True)
            sim_8192 = float(util.cos_sim(emb_as, emb_ls)[0][0])
            
            # 2. NER Recall
            ner_as_ls, ner_ls_as, as_ents, ls_ents = calculate_ner_recall(as_doc, ls_doc)
            
            # 3. Readability
            as_flesch, as_wiener, as_lix = calculate_readability(as_text)
            ls_flesch, ls_wiener, ls_lix = calculate_readability(ls_text)
            
            # 4. Lexical Diversity
            as_tokens, as_ttr, as_mattr = calculate_ttr_metrics(as_doc)
            ls_tokens, ls_ttr, ls_mattr = calculate_ttr_metrics(ls_doc)
            
            # Sentence lengths
            as_sents = list(as_doc.sents)
            ls_sents = list(ls_doc.sents)
            as_avg_sent = len(as_doc) / len(as_sents) if as_sents else 0
            ls_avg_sent = len(ls_doc) / len(ls_sents) if ls_sents else 0
            
            all_rows.append({
                "source": source_name,
                "as_url": as_url,
                "ls_url": ls_url,
                "as_text": as_text,
                "ls_text": ls_text,
                "semantic_similarity_8192": sim_8192,
                "ner_recall_as_ls": ner_as_ls,
                "ner_recall_ls_as": ner_ls_as,
                "as_ent_count": as_ents,
                "ls_ent_count": ls_ents,
                "as_tokens": as_tokens,
                "ls_tokens": ls_tokens,
                "as_avg_sent_len": as_avg_sent,
                "ls_avg_sent_len": ls_avg_sent,
                "as_flesch": as_flesch,
                "as_wiener": as_wiener,
                "as_lix": as_lix,
                "ls_flesch": ls_flesch,
                "ls_wiener": ls_wiener,
                "ls_lix": ls_lix,
                "as_ttr": as_ttr,
                "as_mattr": as_mattr,
                "ls_ttr": ls_ttr,
                "ls_mattr": ls_mattr,
            })
            
    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"Master CSV successfully saved with {len(df)} pairs to: {args.output_csv}")

if __name__ == "__main__":
    main()
