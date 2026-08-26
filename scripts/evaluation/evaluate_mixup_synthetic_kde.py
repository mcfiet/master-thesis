#!/usr/bin/env python3
"""
scripts/evaluation/evaluate_mixup_synthetic_kde.py
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import spacy

nlp = spacy.blank("de")

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

def load_model(model_path, vocab_size, device):
    model = BiLSTMRegressor(vocab_size)
    state = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def tokenize_and_predict(model, vocab, texts, device, max_len=256):
    tokenized = []
    for text in texts:
        doc = nlp(str(text or ""))
        tokens = [t.text.lower() for t in doc if not t.is_space]
        ids = [vocab.get(t, vocab.get("<unk>", 1)) for t in tokens][:max_len]
        if len(ids) == 0:
            ids = [0]
        tokenized.append(ids)
    
    padded = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, seq in enumerate(tokenized):
        padded[i, :len(seq)] = seq
    
    x = torch.tensor(padded, dtype=torch.long, device=device)
    with torch.no_grad():
        scores = model(x)
    return scores.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", default="data/lebenshilfe/lebenshilfe_dataset_clean.json")
    parser.add_argument("--mixup_model_path", default="results/models/bilstm_mixup_regression.pt")
    parser.add_argument("--mixup_vocab_path", default="data/vocabs/mixup_vocab.json")
    parser.add_argument("--synthetic_model_path", default="results/models/bilstm_synthetic_regression.pt")
    parser.add_argument("--synthetic_vocab_path", default="data/vocabs/synthetic_vocab.json")
    parser.add_argument("--output_csv", default="results/evaluation/mixup_synthetic_kde_eval.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.test_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    as_texts = [d.get("as_text") or d.get("source_text") for d in data]
    ls_texts = [d.get("ls_text") or d.get("target_text") for d in data]
    titles = [d.get("source") or d.get("title", f"Sample_{i}") for i, d in enumerate(data)]

    mixup_vocab = load_vocab_dict(args.mixup_vocab_path)
    synth_vocab = load_vocab_dict(args.synthetic_vocab_path)

    mixup_model = load_model(args.mixup_model_path, len(mixup_vocab), args.device)
    synth_model = load_model(args.synthetic_model_path, len(synth_vocab), args.device)

    mixup_as = tokenize_and_predict(mixup_model, mixup_vocab, as_texts, args.device)
    mixup_ls = tokenize_and_predict(mixup_model, mixup_vocab, ls_texts, args.device)

    synth_as = tokenize_and_predict(synth_model, synth_vocab, as_texts, args.device)
    synth_ls = tokenize_and_predict(synth_model, synth_vocab, ls_texts, args.device)

    rows = []
    for i in range(len(data)):
        rows.append({
            "title": titles[i],
            "mixup_as_score": float(mixup_as[i]),
            "mixup_ls_score": float(mixup_ls[i]),
            "mixup_margin": float(mixup_ls[i] - mixup_as[i]),
            "synth_as_score": float(synth_as[i]),
            "synth_ls_score": float(synth_ls[i]),
            "synth_margin": float(synth_ls[i] - synth_as[i])
        })

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"KDE Details gespeichert in {args.output_csv}")

if __name__ == "__main__":
    main()
