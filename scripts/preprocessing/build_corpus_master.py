#!/usr/bin/env python3
"""
Unified Master Corpus Builder & Quality Pipeline:
1. Lädt Rohdaten aus den Quell-JSONs.
2. Wendet die zentrale Reinigung via scripts.data_collection.cleaner an:
   - Tag-Punctuation-Guard
   - Kicker-Stripping ('Sachsen', 'Sachsen-Anhalt', etc.)
   - Bullet-Points ('•', '*', ': •') zu flüssigem Text
   - Normalisierung doppelter Punkte ('..', ': ..') und Satzzeichen
   - Entfernung von Web-Navigation, Autoren- und Prüfer-Credits
3. Filtert nach:
   - Mindestwortanzahl (>= 30 Wörter)
   - Längenverhältnis (0.20 <= LS/AS <= 4.00)
   - Semantischer Ähnlichkeit (0.60 <= Jina SBERT Cosine <= 0.99)
   - Platzhaltern (Lorem Ipsum)
4. Berechnet linguistische und Lesbarkeitsmetriken:
   - SpaCy NER-Recall (AS -> LS, LS -> AS)
   - Lesbarkeit (Flesch Reading Ease, Wiener Sachtextformel, LIX)
   - Lexikalische Diversität (TTR, MATTR)
   - Satzlängen
5. Dedupliziert und speichert:
   - data/analysis/corpus_master.csv
   - data/analysis/corpus_master.json
   - data/corpus/4_normalized_clean/<source>_articles.json
"""

import os
import sys
import json
import glob
import re
import argparse

import pandas as pd
import spacy
import textstat
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch

# Pfad für cleaner.py einbinden
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_collection')))
import cleaner

textstat.set_lang('de')

def parse_args():
    parser = argparse.ArgumentParser(description="Build a unified, cleaned master corpus.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/corpus/2_raw_scraped",
        help="Path to raw scraped source JSON files (fallback to 4_normalized_clean if not found)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/analysis/corpus_master.csv",
        help="Path to save the generated Master CSV."
    )
    parser.add_argument(
        "--clean_json_dir",
        type=str,
        default="data/corpus/4_normalized_clean",
        help="Directory to save cleaned source JSON files."
    )
    parser.add_argument(
        "--sbert_model",
        type=str,
        default="jinaai/jina-embeddings-v2-base-de",
        help="SBERT model for semantic similarity."
    )
    parser.add_argument(
        "--spacy_model",
        type=str,
        default="de_core_news_lg",
        help="SpaCy model for linguistic parsing and NER."
    )
    parser.add_argument("--sim_min", type=float, default=0.60, help="Min SBERT similarity")
    parser.add_argument("--sim_max", type=float, default=0.99, help="Max SBERT similarity")
    parser.add_argument("--min_words", type=int, default=30, help="Min words per text")
    parser.add_argument("--min_ratio", type=float, default=0.20, help="Min LS/AS word ratio")
    parser.add_argument("--max_ratio", type=float, default=4.00, help="Max LS/AS word ratio")
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
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    token_count = len(tokens)
    if token_count == 0:
        return 0, None, None
    
    unique_types = len(set(tokens))
    ttr = unique_types / token_count
    
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

def is_valid_pair(ls_text, as_text, min_words=30, min_ratio=0.20, max_ratio=4.00):
    ls_words = len(ls_text.split())
    as_words = len(as_text.split())
    if ls_words < min_words or as_words < min_words:
        return False
    ratio = ls_words / as_words
    if not (min_ratio <= ratio <= max_ratio):
        return False
    if "lorem ipsum" in ls_text.lower() or "lorem ipsum" in as_text.lower():
        return False
    return True

def main():
    args = parse_args()
    
    input_dir = args.input_dir
    if not os.path.exists(input_dir) or len(glob.glob(os.path.join(input_dir, "*.json"))) == 0:
        print(f"Input directory '{input_dir}' empty or not found. Falling back to 'data/corpus/4_normalized_clean'...")
        input_dir = "data/corpus/4_normalized_clean"
        
    print(f"Loading SpaCy model: {args.spacy_model}...")
    try:
        nlp = spacy.load(args.spacy_model)
    except OSError:
        fallback_models = ["de_core_news_md", "de_core_news_sm"]
        nlp = None
        for model in fallback_models:
            try:
                nlp = spacy.load(model)
                print(f"Loaded fallback {model}.")
                break
            except OSError:
                continue
        if nlp is None:
            raise RuntimeError("Could not load any German SpaCy model.")
            
    nlp.max_length = 2000000
    
    print(f"Loading SentenceTransformer: {args.sbert_model}...")
    sbert = SentenceTransformer(args.sbert_model, trust_remote_code=True)
    sbert.max_seq_length = 8192
    
    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    print(f"Found {len(json_files)} source files in '{input_dir}'.")
    
    all_rows = []
    seen_pairs = set()
    cleaned_source_data = {}
    
    for file_path in json_files:
        source_name = os.path.basename(file_path).replace("_articles.json", "")
        print(f"Processing source: {source_name}...")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            pairs = data.get("pairs", []) if isinstance(data, dict) else data
            
        cleaned_source_data[source_name] = []
        
        for pair in tqdm(pairs, desc=source_name):
            raw_as = pair.get("as_text", "").strip()
            if not raw_as and "as_texts" in pair and isinstance(pair["as_texts"], list):
                raw_as = " ".join(pair["as_texts"]).strip()
            raw_ls = pair.get("ls_text", "").strip()
            if not raw_as or not raw_ls:
                continue
                
            # 1. Cleaner Transformation
            clean_ls, clean_as = cleaner.clean_pair(raw_ls, raw_as, source=source_name)
            
            # 2. Length & Placeholder Validation
            if not is_valid_pair(clean_ls, clean_as, args.min_words, args.min_ratio, args.max_ratio):
                continue
                
            # 3. Deduplication
            dedup_key = (clean_ls, clean_as)
            if dedup_key in seen_pairs:
                continue
            seen_pairs.add(dedup_key)
            
            # 4. SBERT Similarity Filter
            with torch.inference_mode():
                emb_as = sbert.encode(clean_as, convert_to_tensor=True, show_progress_bar=False)
                emb_ls = sbert.encode(clean_ls, convert_to_tensor=True, show_progress_bar=False)
                sim_8192 = float(util.cos_sim(emb_as, emb_ls)[0][0].item())
                del emb_as, emb_ls
            
            if not (args.sim_min <= sim_8192 <= args.sim_max):
                continue
                
            as_url = pair.get("as_url") or pair.get("url") or "unknown"
            ls_url = pair.get("ls_url") or "unknown"
            
            # 5. SpaCy NLP & Metrics
            as_doc = nlp(clean_as)
            ls_doc = nlp(clean_ls)
            
            ner_as_ls, ner_ls_as, as_ents, ls_ents = calculate_ner_recall(as_doc, ls_doc)
            as_flesch, as_wiener, as_lix = calculate_readability(clean_as)
            ls_flesch, ls_wiener, ls_lix = calculate_readability(clean_ls)
            as_tokens, as_ttr, as_mattr = calculate_ttr_metrics(as_doc)
            ls_tokens, ls_ttr, ls_mattr = calculate_ttr_metrics(ls_doc)
            
            as_sents = list(as_doc.sents)
            ls_sents = list(ls_doc.sents)
            as_avg_sent = len(as_doc) / len(as_sents) if as_sents else 0
            ls_avg_sent = len(ls_doc) / len(ls_sents) if ls_sents else 0
            
            del as_doc, ls_doc
            
            row_dict = {
                "source": source_name,
                "as_url": as_url,
                "ls_url": ls_url,
                "as_text": clean_as,
                "ls_text": clean_ls,
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
            }
            all_rows.append(row_dict)
            
            cleaned_source_data[source_name].append({
                "as_url": as_url,
                "ls_url": ls_url,
                "as_text": clean_as,
                "ls_text": clean_ls,
                "ls_tokens": ls_tokens,
                "as_tokens": as_tokens,
                "semantic_similarity_8192": sim_8192
            })
            
    # Save Master CSV and JSON
    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"Saved Master CSV with {len(df)} pairs to: {args.output_csv}")
    
    output_json = args.output_csv.replace(".csv", ".json")
    df.to_json(output_json, orient="records", force_ascii=False, indent=2)
    print(f"Saved Master JSON with {len(df)} pairs to: {output_json}")
    
    # Save clean source JSONs
    os.makedirs(args.clean_json_dir, exist_ok=True)
    for src, src_pairs in cleaned_source_data.items():
        src_file = os.path.join(args.clean_json_dir, f"{src}_articles.json")
        with open(src_file, "w", encoding="utf-8") as f:
            json.dump({"source": src, "count": len(src_pairs), "pairs": src_pairs}, f, ensure_ascii=False, indent=4)
    print(f"Saved individual cleaned source JSONs to: {args.clean_json_dir}")

if __name__ == "__main__":
    main()
