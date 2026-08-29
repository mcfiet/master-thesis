#!/usr/bin/env python3
"""
scripts/experiments/classifier_stability/train_and_track_stability.py

Trainiert systematisch mehrere Klassifikations- und Regressionsmodelle über verschiedene
Zufalls-Seeds (z.B. 42, 123, 456, 789, 1024) sowie Modellkapazitäten (Full, Medium, Tiny).
Trackt nach JEDER Epoche:
1. In-Domain Train & Validation Loss / Balanced Accuracy
2. Out-of-Domain Lebenshilfe Goldstandard Balanced Accuracy, ROC-AUC, Separation & Pair-Match

Speichert:
- Modell-Checkpoints in results/models/classifier_stability/
- Epochen-Trajektorien in results/evaluation/classifier_stability/epoch_trajectories.json
"""

import os
import sys
import json
import random
import argparse
from collections import Counter
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
from tqdm import tqdm


# ==============================================================================
# MODEL ARCHITECTURES
# ==============================================================================
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(self.dropout(hidden))


class BiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.sigmoid(self.fc(self.dropout(hidden)))


# ==============================================================================
# VOCAB & DATASETS
# ==============================================================================
class Vocab:
    def __init__(self, tokenized_texts: List[List[str]], max_size: int = 25000, min_freq: int = 3):
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
        self.itos = ["<pad>", "<unk>"]
        self.stoi = {"<pad>": 0, "<unk>": 1}
        for token, freq in counter.most_common(max_size):
            if freq >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]


class TextDataset(Dataset):
    def __init__(self, X: List[List[str]], y: List[float], vocab: Vocab, max_len: int):
        self.X = X
        self.y = y
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        tokens = self.X[idx]
        encoded = self.vocab.encode(tokens)[:self.max_len]
        padded = encoded + [0] * (self.max_len - len(encoded))
        return torch.tensor(padded, dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# DATA LOADERS
# ==============================================================================
def load_corpus_articles(csv_path: str, min_sim: float = 0.80, max_sim: float = 0.98) -> Tuple[List[List[str]], List[int]]:
    print(f"[DATA] Lade Artikel aus {csv_path} (Filter: {min_sim} <= sim <= {max_sim})...")
    df = pd.read_csv(csv_path)
    mask = (df["semantic_similarity_8192"] >= min_sim) & (df["semantic_similarity_8192"] <= max_sim)
    df_filtered = df[mask]
    print(f"[DATA] {len(df_filtered)} Artikelpaare gefunden.")

    nlp = spacy.blank("de")
    X, y = [], []
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Tokenisiere Artikel"):
        ls_text, as_text = str(row["ls_text"]), str(row["as_text"])
        ls_tokens = [t.text.lower() for t in nlp(ls_text) if not t.is_space]
        if len(ls_tokens) >= 10:
            X.append(ls_tokens)
            y.append(1)
        as_tokens = [t.text.lower() for t in nlp(as_text) if not t.is_space]
        if len(as_tokens) >= 10:
            X.append(as_tokens)
            y.append(0)
    print(f"[DATA] Geladen: {len(X)} Dokumente ({y.count(1)} LS, {y.count(0)} AS)")
    return X, y


def load_corpus_sentences(csv_path: str, min_sim: float = 0.80, max_sim: float = 0.98, max_pairs: int = 1500) -> Tuple[List[List[str]], List[int]]:
    print(f"[DATA] Lade Einzelsätze aus {csv_path}...")
    df = pd.read_csv(csv_path)
    mask = (df["semantic_similarity_8192"] >= min_sim) & (df["semantic_similarity_8192"] <= max_sim)
    df_filtered = df[mask].head(max_pairs)

    nlp = spacy.blank("de")
    nlp.add_pipe("sentencizer")

    X, y = [], []
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Satzextraktion"):
        doc_ls = nlp(str(row["ls_text"]))
        for sent in doc_ls.sents:
            tokens = [t.text.lower() for t in sent if not t.is_space]
            if len(tokens) >= 3:
                X.append(tokens)
                y.append(1)
        doc_as = nlp(str(row["as_text"]))
        for sent in doc_as.sents:
            tokens = [t.text.lower() for t in sent if not t.is_space]
            if len(tokens) >= 3:
                X.append(tokens)
                y.append(0)
    print(f"[DATA] Geladen: {len(X)} Sätze ({y.count(1)} LS, {y.count(0)} AS)")
    return X, y


def load_lebenshilfe_goldstandard(lh_path: str) -> List[Dict[str, Any]]:
    with open(lh_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nlp = spacy.blank("de")
    nlp.add_pipe("sentencizer")
    clean_items = []
    for item in data:
        ls_text = item.get("ls_text", "")
        as_text = item.get("as_text", "")
        ls_tokens = [t.text.lower() for t in nlp(ls_text) if not t.is_space]
        as_tokens = [t.text.lower() for t in nlp(as_text) if not t.is_space]
        ls_sents = [[t.text.lower() for t in s if not t.is_space] for s in nlp(ls_text).sents if len(s) > 0]
        as_sents = [[t.text.lower() for t in s if not t.is_space] for s in nlp(as_text).sents if len(s) > 0]
        clean_items.append({
            "pair_id": item.get("pair_id", 0),
            "ls_tokens": ls_tokens,
            "as_tokens": as_tokens,
            "ls_sents": ls_sents,
            "as_sents": as_sents,
        })
    return clean_items


# ==============================================================================
# OOD EVALUATION HELPER
# ==============================================================================
def evaluate_on_lebenshilfe(
    model: nn.Module,
    vocab: Vocab,
    lh_items: List[Dict[str, Any]],
    max_len: int,
    is_regressor: bool,
    is_sentence_model: bool,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    y_true, y_scores = [], []
    pair_correct = 0

    with torch.no_grad():
        for item in lh_items:
            if is_sentence_model:
                # Majority Voting über Sätze
                ls_sent_scores = []
                for s_tokens in item["ls_sents"]:
                    enc = vocab.encode(s_tokens)[:128]
                    pad = enc + [0] * (128 - len(enc))
                    bx = torch.tensor([pad], dtype=torch.long, device=device)
                    out = model(bx).squeeze(-1)
                    score = torch.sigmoid(out).item() if not is_regressor else out.item()
                    ls_sent_scores.append(1 if score >= 0.5 else 0)
                ls_score = np.mean(ls_sent_scores) if ls_sent_scores else 0.5

                as_sent_scores = []
                for s_tokens in item["as_sents"]:
                    enc = vocab.encode(s_tokens)[:128]
                    pad = enc + [0] * (128 - len(enc))
                    bx = torch.tensor([pad], dtype=torch.long, device=device)
                    out = model(bx).squeeze(-1)
                    score = torch.sigmoid(out).item() if not is_regressor else out.item()
                    as_sent_scores.append(1 if score >= 0.5 else 0)
                as_score = np.mean(as_sent_scores) if as_sent_scores else 0.5
            else:
                # Dokument-Inferenz
                enc_ls = vocab.encode(item["ls_tokens"])[:max_len]
                pad_ls = enc_ls + [0] * (max_len - len(enc_ls))
                bx_ls = torch.tensor([pad_ls], dtype=torch.long, device=device)
                out_ls = model(bx_ls).squeeze(-1)
                ls_score = torch.sigmoid(out_ls).item() if not is_regressor else out_ls.item()

                enc_as = vocab.encode(item["as_tokens"])[:max_len]
                pad_as = enc_as + [0] * (max_len - len(enc_as))
                bx_as = torch.tensor([pad_as], dtype=torch.long, device=device)
                out_as = model(bx_as).squeeze(-1)
                as_score = torch.sigmoid(out_as).item() if not is_regressor else out_as.item()

            y_true.extend([1, 0])
            y_scores.extend([ls_score, as_score])
            if ls_score > as_score:
                pair_correct += 1

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= 0.5).astype(int)

    bacc = float(balanced_accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) > 1 else 0.5
    ls_mean = float(np.mean(y_scores[y_true == 1]))
    as_mean = float(np.mean(y_scores[y_true == 0]))
    separation = float(ls_mean - as_mean)
    pair_match = float(pair_correct / len(lh_items))

    return {
        "lh_bacc": bacc,
        "lh_auc": auc,
        "lh_ls_mean": ls_mean,
        "lh_as_mean": as_mean,
        "lh_separation": separation,
        "lh_pair_match": pair_match,
    }


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def run_stability_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "vocabs"), exist_ok=True)

    # 1. Daten laden
    article_X, article_y = load_corpus_articles(args.csv_path, min_sim=args.min_sim, max_sim=args.max_sim)
    sentence_X, sentence_y = load_corpus_sentences(args.csv_path, min_sim=args.min_sim, max_sim=args.max_sim)
    lh_items = load_lebenshilfe_goldstandard(args.lh_dataset_path)

    # Modellkonfigurationen
    model_configs = {
        "art_256": {"type": "classifier", "data": "article", "max_len": 256, "vocab_size": 25000, "embed_dim": 128, "hidden_dim": 128},
        "art_512": {"type": "classifier", "data": "article", "max_len": 512, "vocab_size": 25000, "embed_dim": 128, "hidden_dim": 128},
        "art_1024": {"type": "classifier", "data": "article", "max_len": 1024, "vocab_size": 25000, "embed_dim": 128, "hidden_dim": 128},
        "art_1024_tiny": {"type": "classifier", "data": "article", "max_len": 1024, "vocab_size": 5000, "embed_dim": 32, "hidden_dim": 32},
        "art_1024_medium": {"type": "classifier", "data": "article", "max_len": 1024, "vocab_size": 10000, "embed_dim": 64, "hidden_dim": 64},
        "sentence_model": {"type": "classifier", "data": "sentence", "max_len": 128, "vocab_size": 25000, "embed_dim": 128, "hidden_dim": 128},
        "mixup_1024": {"type": "regressor", "data": "article", "max_len": 1024, "vocab_size": 25000, "embed_dim": 128, "hidden_dim": 128},
    }

    if args.models:
        model_configs = {k: v for k, v in model_configs.items() if k in args.models}

    trajectories = {}
    summary_results = []

    for model_name, cfg in model_configs.items():
        print("\n" + "=" * 80)
        print(f"STARTE MODELLKONFIGURATION: {model_name} (Type: {cfg['type']}, Vocab: {cfg['vocab_size']}, Hidden: {cfg['hidden_dim']})")
        print("=" * 80)

        trajectories[model_name] = {}

        for seed in args.seeds:
            print(f"\n---> Trainiere {model_name} mit Seed {seed}...")
            set_seed(seed)

            # Split vorbereiten
            data_X = article_X if cfg["data"] == "article" else sentence_X
            data_y = article_y if cfg["data"] == "article" else sentence_y

            X_tv, X_test, y_tv, y_test = train_test_split(data_X, data_y, test_size=0.15, random_state=seed, stratify=data_y)
            X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.15, random_state=seed, stratify=y_tv)

            vocab = Vocab(X_train, max_size=cfg["vocab_size"], min_freq=3)

            train_ds = TextDataset(X_train, y_train, vocab, cfg["max_len"])
            val_ds = TextDataset(X_val, y_val, vocab, cfg["max_len"])

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

            is_reg = (cfg["type"] == "regressor")
            is_sent = (cfg["data"] == "sentence")

            if is_reg:
                model = BiLSTMRegressor(len(vocab), cfg["embed_dim"], cfg["hidden_dim"]).to(device)
                criterion = nn.MSELoss()
            else:
                model = BiLSTMClassifier(len(vocab), cfg["embed_dim"], cfg["hidden_dim"]).to(device)
                criterion = nn.BCEWithLogitsLoss()

            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

            epoch_history = []
            best_val_bacc = 0.0
            best_ckpt_metrics = None
            save_path = os.path.join(args.output_dir, "models", f"{model_name}_seed_{seed}.pt")

            for epoch in range(1, args.epochs + 1):
                model.train()
                train_loss_acc = 0.0
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    out = model(bx).squeeze(-1)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    train_loss_acc += loss.item()

                train_loss = train_loss_acc / len(train_loader)

                # In-Domain Validation
                model.eval()
                val_loss_acc = 0.0
                val_preds, val_targets = [], []
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(device), by.to(device)
                        out = model(bx).squeeze(-1)
                        val_loss_acc += criterion(out, by).item()
                        score = torch.sigmoid(out) if not is_reg else out
                        val_preds.extend((score >= 0.5).cpu().numpy().astype(int))
                        val_targets.extend(by.cpu().numpy().astype(int))

                val_loss = val_loss_acc / len(val_loader)
                val_bacc = float(balanced_accuracy_score(val_targets, val_preds))

                # Out-of-Domain Evaluation (Lebenshilfe Goldstandard)
                ood_metrics = evaluate_on_lebenshilfe(
                    model, vocab, lh_items, cfg["max_len"], is_reg, is_sent, device
                )

                epoch_stat = {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_bacc": float(val_bacc),
                    **ood_metrics
                }
                epoch_history.append(epoch_stat)

                if val_bacc > best_val_bacc:
                    best_val_bacc = val_bacc
                    best_ckpt_metrics = epoch_stat.copy()
                    torch.save(model.state_dict(), save_path)

                if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
                    print(f"  Ep {epoch:2d}/{args.epochs} | TrainLoss: {train_loss:.4f} | InVal-BAcc: {val_bacc*100:5.2f}% | OOD-BAcc: {ood_metrics['lh_bacc']*100:5.2f}% | OOD-Sep: {ood_metrics['lh_separation']:.3f}")

            trajectories[model_name][str(seed)] = epoch_history
            if best_ckpt_metrics is not None:
                summary_results.append({
                    "model": model_name,
                    "seed": seed,
                    **best_ckpt_metrics
                })

    # Trajektorien & Zusammenfassung speichern
    traj_path = os.path.join(args.output_dir, "epoch_trajectories.json")
    with open(traj_path, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=2)
    print(f"\n[OK] Epochen-Trajektorien gespeichert in: {traj_path}")

    summary_df = pd.DataFrame(summary_results)
    summary_csv = os.path.join(args.output_dir, "seed_summary_raw.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[OK] Roh-Zusammenfassung gespeichert in: {summary_csv}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed & Capacity Stability Experiment")
    parser.add_argument("--csv_path", default="data/analysis/corpus_master.csv")
    parser.add_argument("--lh_dataset_path", default="data/lebenshilfe/lebenshilfe_dataset_clean.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--min_sim", type=float, default=0.80)
    parser.add_argument("--max_sim", type=float, default=0.98)
    parser.add_argument("--models", nargs="+", default=None, help="Specific models to train (or all by default)")
    parser.add_argument("--output_dir", default="results/evaluation/classifier_stability")
    args = parser.parse_args()

    run_stability_experiment(args)


if __name__ == "__main__":
    main()
