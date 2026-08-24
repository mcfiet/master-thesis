#!/usr/bin/env python3
"""
scripts/evaluation/evaluate_textcomplexityde.py
"""
import os
import sys
import json
import argparse
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class BiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, dropout=0.3):
        super(BiLSTMRegressor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(self.dropout(hidden))
        return self.sigmoid(out).squeeze(-1)

def load_vocab_dict(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "stoi" in data:
        return data["stoi"]
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="results/models/bilstm_mixup_regression.pt")
    parser.add_argument("--vocab_path", default="data/vocabs/mixup_vocab.json")
    parser.add_argument("--benchmark_csv", default="data/analysis/textcomplexityde/ratings.csv")
    parser.add_argument("--output_csv", default="results/evaluation/textcomplexityde_eval.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.path.exists(args.benchmark_csv):
        os.makedirs(os.path.dirname(args.benchmark_csv), exist_ok=True)
        url = "https://raw.githubusercontent.com/babaknaderi/TextComplexityDE/master/data/ratings.csv"
        urllib.request.urlretrieve(url, args.benchmark_csv)

    try:
        df_raw = pd.read_csv(args.benchmark_csv, encoding="utf-8")
    except UnicodeDecodeError:
        df_raw = pd.read_csv(args.benchmark_csv, encoding="latin-1")

    if "Sentence" in df_raw.columns and "MOS_Complexity" in df_raw.columns:
        df_sentences = df_raw.groupby("Sentence")[["MOS_Complexity", "MOS_Understandability", "MOS_Lexical_difficulty"]].mean().reset_index()
    elif "Sentence" in df_raw.columns and "Complexity" in df_raw.columns:
        df_sentences = df_raw.groupby("Sentence")["Complexity"].mean().reset_index().rename(columns={"Complexity": "MOS_Complexity"})
        df_sentences["MOS_Understandability"] = df_sentences["MOS_Complexity"]
        df_sentences["MOS_Lexical_difficulty"] = df_sentences["MOS_Complexity"]
    else:
        df_sentences = df_raw

    vocab = load_vocab_dict(args.vocab_path)
    model = BiLSTMRegressor(len(vocab))
    state = torch.load(args.model_path, map_location=args.device, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    sentences = df_sentences["Sentence"].tolist()
    y_human_raw = df_sentences["MOS_Complexity"].to_numpy()
    y_human_simplicity = 1.0 - ((y_human_raw - y_human_raw.min()) / (y_human_raw.max() - y_human_raw.min()))

    tokenized = []
    for s in sentences:
        tokens = str(s).lower().split()
        ids = [vocab.get(t, vocab.get("<unk>", 1)) for t in tokens][:256]
        if len(ids) == 0:
            ids = [0]
        tokenized.append(ids)

    padded = np.zeros((len(sentences), 256), dtype=np.int64)
    for i, seq in enumerate(tokenized):
        padded[i, :len(seq)] = seq

    x = torch.tensor(padded, dtype=torch.long, device=args.device)
    with torch.no_grad():
        preds = model(x).cpu().numpy()

    df_sentences["Human_Simplicity_Norm"] = y_human_simplicity
    df_sentences["pred_simplicity"] = preds

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_sentences.to_csv(args.output_csv, index=False)
    print(f"Ergebnisse ({len(df_sentences)} Sätze) erfolgreich gespeichert in {args.output_csv}")

    # Print Summary Metrics
    pearson_r = float(pearsonr(preds, df_sentences["MOS_Complexity"])[0])
    spearman_rho = float(spearmanr(preds, df_sentences["MOS_Complexity"])[0])
    print(f"\n--- TextComplexityDE Benchmark Zusammenfassung ---")
    print(f"Pearson r vs. MOS Komplexität:   {pearson_r:.4f}")
    print(f"Spearman rho vs. MOS Komplexität: {spearman_rho:.4f}")
    print(f"R² Score auf Simplicity:          {float(r2_score(y_human_simplicity, preds)):.4f}")

if __name__ == "__main__":
    main()

