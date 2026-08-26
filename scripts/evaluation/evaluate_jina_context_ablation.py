#!/usr/bin/env python3
"""
Context Length Ablation Experiment (Jina Embeddings v2):
Vergleicht die semantische Ähnlichkeit (Kosinus-Ähnlichkeit) zwischen AS und LS
bei variierenden Kontextfenstern (128, 256, 512, 1024, 8192 Tokens) zur
systematischen Untersuchung von Truncation-Artefakten.
"""

import os
import sys
import json
import argparse
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util

def parse_args():
    parser = argparse.ArgumentParser(description="Run Jina context length ablation experiment.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/analysis/corpus_master.json",
        help="Path to corpus_master.json or directory of cleaned JSON files."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/analysis/jina_context_ablation.csv",
        help="Output CSV path for detailed pair-level results."
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="results/evaluation/jina_context_ablation_summary.csv",
        help="Output CSV path for aggregated source-level summary."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="jinaai/jina-embeddings-v2-base-de",
        help="HuggingFace model identifier for long-context embeddings."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference batch size."
    )
    return parser.parse_args()

def load_data(input_path):
    if os.path.isfile(input_path):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "pairs" in data:
            return data["pairs"]
    elif os.path.isdir(input_path):
        pairs = []
        for fp in sorted(glob.glob(os.path.join(input_path, "*.json"))):
            source = os.path.basename(fp).replace("_articles.json", "")
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
            src_pairs = d.get("pairs", []) if isinstance(d, dict) else d
            for p in src_pairs:
                p["source"] = p.get("source", source)
                pairs.append(p)
        return pairs
    raise ValueError(f"Could not load data from {input_path}")

def main():
    args = parse_args()
    print(f"=== Starte Kontextlängen-Ablation mit Modell: {args.model_name} ===")
    
    pairs = load_data(args.input_path)
    print(f"Datensatz geladen: {len(pairs)} Artikelpaare gefunden.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Verwende Device: {device}")
    
    sbert = SentenceTransformer(args.model_name, trust_remote_code=True, device=device)
    
    context_lengths = [128, 256, 512, 1024, 8192]
    results = []
    
    for pair in tqdm(pairs, desc="Berechne Kontext-Ähnlichkeiten"):
        source = pair.get("source", "unknown")
        as_text = pair.get("as_text", "").strip()
        ls_text = pair.get("ls_text", "").strip()
        
        if not as_text or not ls_text:
            continue
            
        row = {
            "source": source,
            "as_url": pair.get("as_url", ""),
            "ls_url": pair.get("ls_url", ""),
            "as_word_count": len(as_text.split()),
            "ls_word_count": len(ls_text.split()),
        }
        
        for ctx in context_lengths:
            sbert.max_seq_length = ctx
            emb_as = sbert.encode(as_text, convert_to_tensor=True, device=device)
            emb_ls = sbert.encode(ls_text, convert_to_tensor=True, device=device)
            sim = float(util.cos_sim(emb_as, emb_ls)[0][0].item())
            row[f"sim_{ctx}"] = sim
            
        results.append(row)
        
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"Detaillierte Paar-Ergebnisse gespeichert in: {args.output_csv}")
    
    # Aggregierte Zusammenfassung pro Quelle
    sim_cols = [f"sim_{ctx}" for ctx in context_lengths]
    summary = df.groupby("source")[sim_cols].agg(["mean", "std"]).reset_index()
    
    # Flache Spaltennamen
    summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns]
    if "sim_8192_mean" in summary.columns and "sim_128_mean" in summary.columns:
        summary["delta_8192_minus_128"] = summary["sim_8192_mean"] - summary["sim_128_mean"]
        
    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
    summary.to_csv(args.summary_csv, index=False, encoding="utf-8")
    print(f"Aggregierte Zusammenfassung gespeichert in: {args.summary_csv}")
    
    print("\n=== Übersicht der Mittelwerte nach Quelle ===")
    display_cols = ["source"] + [f"sim_{ctx}_mean" for ctx in [128, 512, 8192] if f"sim_{ctx}_mean" in summary.columns] + ["delta_8192_minus_128"]
    print(summary[display_cols].to_string(index=False))

if __name__ == "__main__":
    main()
