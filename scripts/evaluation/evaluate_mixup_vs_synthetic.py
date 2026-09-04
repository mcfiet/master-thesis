#!/usr/bin/env python3
"""
scripts/evaluation/evaluate_mixup_vs_synthetic.py
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

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
    scores = []
    with torch.no_grad():
        for text in texts:
            doc = nlp(str(text or ""))
            tokens = [t.text.lower() for t in doc if not t.is_space]
            ids = [vocab.get(t, vocab.get("<unk>", 1)) for t in tokens][:max_len]
            if len(ids) == 0:
                ids = [0]
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            p = model(inp).squeeze().item()
            scores.append(p)
    return np.array(scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixup_model_path", default="results/models/bilstm_mixup_regression.pt")
    parser.add_argument("--mixup_vocab_path", default="data/vocabs/mixup_vocab.json")
    parser.add_argument("--synthetic_model_path", default="results/models/bilstm_synthetic_regression.pt")
    parser.add_argument("--synthetic_vocab_path", default="data/vocabs/synthetic_vocab.json")
    parser.add_argument("--steps_dataset_path", default="data/lebenshilfe/lebenshilfe_dataset_with_steps.json")
    parser.add_argument("--output_csv", default="results/evaluation/mixup_vs_synthetic_unbiased_eval.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    mixup_vocab = load_vocab_dict(args.mixup_vocab_path)
    synth_vocab = load_vocab_dict(args.synthetic_vocab_path)

    mixup_model = load_model(args.mixup_model_path, len(mixup_vocab), args.device)
    synth_model = load_model(args.synthetic_model_path, len(synth_vocab), args.device)

    with open(args.steps_dataset_path, "r", encoding="utf-8") as f:
        steps_data = json.load(f)

    texts_steps, y_true_steps = [], []
    for item in steps_data:
        steps = item.get("intermediate_steps", item.get("simplification_steps", []))
        if len(steps) >= 5:
            targets = [0.0, 0.25, 0.5, 0.75, 1.0]
            for step_txt, tgt in zip(steps[:5], targets):
                texts_steps.append(step_txt)
                y_true_steps.append(tgt)

    y_true_steps = np.array(y_true_steps)
    preds_mixup = tokenize_and_predict(mixup_model, mixup_vocab, texts_steps, args.device)
    preds_synth = tokenize_and_predict(synth_model, synth_vocab, texts_steps, args.device)

    results = [
        {
            "dataset": "LLM Synthetic Steps",
            "model": "MixUp Regressor",
            "mae": float(mean_absolute_error(y_true_steps, preds_mixup)),
            "mse": float(mean_squared_error(y_true_steps, preds_mixup)),
            "r2": float(r2_score(y_true_steps, preds_mixup)),
            "pearson": float(pearsonr(y_true_steps, preds_mixup)[0]),
            "spearman": float(spearmanr(y_true_steps, preds_mixup)[0])
        },
        {
            "dataset": "LLM Synthetic Steps",
            "model": "Synthetic Regressor",
            "mae": float(mean_absolute_error(y_true_steps, preds_synth)),
            "mse": float(mean_squared_error(y_true_steps, preds_synth)),
            "r2": float(r2_score(y_true_steps, preds_synth)),
            "pearson": float(pearsonr(y_true_steps, preds_synth)[0]),
            "spearman": float(spearmanr(y_true_steps, preds_synth)[0])
        }
    ]

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Gespeichert in {args.output_csv}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
