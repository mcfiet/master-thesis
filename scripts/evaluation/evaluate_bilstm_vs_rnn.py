#!/usr/bin/env python3
"""
scripts/evaluation/evaluate_bilstm_vs_rnn.py
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
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

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

class VanillaRNNRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, dropout=0.3):
        super(VanillaRNNRegressor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, hidden = self.rnn(embedded)
        out = self.fc(self.dropout(hidden[-1, :, :]))
        return self.sigmoid(out).squeeze(-1)

def load_vocab_dict(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "stoi" in data:
        return data["stoi"]
    return data

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
    parser.add_argument("--test_data_path", default="data/lebenshilfe/lebenshilfe_dataset_clean.json")
    parser.add_argument("--bilstm_model_path", default="results/models/bilstm_mixup_regression.pt")
    parser.add_argument("--rnn_model_path", default="results/models/rnn_vanilla_mixup_regression.pt")
    parser.add_argument("--vocab_path", default="data/vocabs/mixup_vocab.json")
    parser.add_argument("--output_csv", default="results/evaluation/bilstm_vs_rnn_eval.csv")
    parser.add_argument("--output_predictions_csv", default="results/evaluation/bilstm_vs_rnn_predictions.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    vocab = load_vocab_dict(args.vocab_path)
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    as_texts = [d.get("as_text") or d.get("source_text") for d in data]
    ls_texts = [d.get("ls_text") or d.get("target_text") for d in data]
    titles = [d.get("source") or d.get("title") or d.get("ls_filename") or f"Sample_{i}" for i, d in enumerate(data)]
    y_true = np.array([0] * len(as_texts) + [1] * len(ls_texts))

    eval_configs = [
        ("BiLSTM (MixUp)", BiLSTMRegressor(len(vocab)), args.bilstm_model_path, "bilstm"),
        ("Vanilla RNN (Baseline)", VanillaRNNRegressor(len(vocab)), args.rnn_model_path, "rnn")
    ]

    results = []
    preds_dict = {
        "sample_idx": list(range(len(data))),
        "source": titles
    }

    for model_name, model, path, key in eval_configs:
        if not os.path.exists(path):
            continue
        state = torch.load(path, map_location=args.device, weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.to(args.device)
        model.eval()

        as_scores = tokenize_and_predict(model, vocab, as_texts, args.device)
        ls_scores = tokenize_and_predict(model, vocab, ls_texts, args.device)
        all_preds = np.concatenate([as_scores, ls_scores])
        binary_preds = (all_preds >= 0.5).astype(int)

        preds_dict[f"as_pred_{key}"] = [float(s) for s in as_scores]
        preds_dict[f"ls_pred_{key}"] = [float(s) for s in ls_scores]
        preds_dict[f"margin_{key}"] = [float(l - a) for l, a in zip(ls_scores, as_scores)]

        results.append({
            "model": model_name,
            "mean_as_score": float(np.mean(as_scores)),
            "mean_ls_score": float(np.mean(ls_scores)),
            "separation_margin": float(np.mean(ls_scores) - np.mean(as_scores)),
            "roc_auc": float(roc_auc_score(y_true, all_preds)),
            "accuracy": float(accuracy_score(y_true, binary_preds)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, binary_preds)),
            "wasserstein_distance": float(wasserstein_distance(as_scores, ls_scores)),
            "ks_statistic": float(ks_2samp(as_scores, ls_scores)[0])
        })

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_summary = pd.DataFrame(results)
    df_summary.to_csv(args.output_csv, index=False)
    print(f"Aggregierte Metriken gespeichert in {args.output_csv}")

    if args.output_predictions_csv:
        os.makedirs(os.path.dirname(args.output_predictions_csv), exist_ok=True)
        df_preds = pd.DataFrame(preds_dict)
        df_preds.to_csv(args.output_predictions_csv, index=False)
        print(f"Sample-Vorhersagen gespeichert in {args.output_predictions_csv}")

if __name__ == "__main__":
    main()
