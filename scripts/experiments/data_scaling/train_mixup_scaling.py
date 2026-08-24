#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Experiment: Data Scaling & Learning Curve Analysis for MixUp Metric Model
=============================================================================
Systematically trains the BiLSTM MixUp Simplicity Regressor across varying
data scale dimensions:
  1. mixtures_per_pair (e.g. 2, 5, 10, 20, 40, 80)
  2. train_fraction / article count (e.g. 10%, 25%, 50%, 75%, 100%)

Evaluates each configuration on a strictly fixed, held-out Test-Split to
generate empirical learning curves for the master thesis.
=============================================================================
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("DataScalingExperiment")


def set_seed(seed: int = 42):
    """Sets global seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Vocab:
    """Vocabulary mapper mapping tokens to integer indices."""
    def __init__(self, tokens: List[str], max_size: int = 25000, min_freq: int = 2):
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = {0: "<pad>", 1: "<unk>"}
        
        freqs: Dict[str, int] = {}
        for t in tokens:
            freqs[t] = freqs.get(t, 0) + 1
            
        sorted_tokens = sorted(
            [t for t, c in freqs.items() if c >= min_freq and t not in self.stoi],
            key=lambda t: freqs[t],
            reverse=True
        )
        
        for t in sorted_tokens:
            if len(self.stoi) >= max_size:
                break
            idx = len(self.stoi)
            self.stoi[t] = idx
            self.itos[idx] = t
            
    def __len__(self):
        return len(self.stoi)
        
    def encode(self, tokens: List[str]) -> List[int]:
        unk_idx = self.stoi["<unk>"]
        return [self.stoi.get(t, unk_idx) for t in tokens]


class MixupPyTorchDataset(Dataset):
    """
    Hybrid MixUp Dataset:
    Pre-generates static baseline mixtures and progressively shifts towards
    dynamic on-the-fly resampling with probability p_dynamic = epoch / total_epochs.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocab,
        nlp_sentencizer: Any,
        max_seq_len: int = 256,
        mixtures_per_pair: int = 20,
        total_epochs: int = 40,
        is_train: bool = True,
        seed: int = 42
    ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.is_train = is_train
        self.current_epoch = 0
        self.total_epochs = total_epochs
        
        self.ls_data: List[List[Tuple[List[str], int]]] = []
        self.as_data: List[List[Tuple[List[str], int]]] = []
        self.static_samples: List[Tuple[int, List[int], float]] = []
        
        split_seed = seed if is_train else 99
        set_seed(split_seed)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Segmenting Sentences", leave=False):
            ls_doc = nlp_sentencizer(str(row["ls_text"]))
            as_doc = nlp_sentencizer(str(row["as_text"]))
            
            ls_sents = []
            for sent in ls_doc.sents:
                text = sent.text.strip()
                tokens = [t.text.lower() for t in sent if not t.is_space]
                if len(tokens) > 0:
                    ls_sents.append((tokens, len(text)))
                    
            as_sents = []
            for sent in as_doc.sents:
                text = sent.text.strip()
                tokens = [t.text.lower() for t in sent if not t.is_space]
                if len(tokens) > 0:
                    as_sents.append((tokens, len(text)))
                    
            num_leicht = len(ls_sents)
            num_alltag = len(as_sents)
            
            if num_leicht == 0 or num_alltag == 0:
                continue
                
            self.ls_data.append(ls_sents)
            self.as_data.append(as_sents)
            
            article_idx = len(self.ls_data) - 1
            for _ in range(mixtures_per_pair):
                start_l, end_l = sorted([random.randint(0, num_leicht), random.randint(0, num_leicht)])
                sample_l = ls_sents[start_l:end_l]
                
                start_a, end_a = sorted([random.randint(0, num_alltag), random.randint(0, num_alltag)])
                sample_a = as_sents[start_a:end_a]
                
                if len(sample_l) == 0 and len(sample_a) == 0:
                    regression_target = 0.5
                    encoded = [0] * self.max_seq_len
                else:
                    char_len_l = sum(item[1] for item in sample_l)
                    char_len_a = sum(item[1] for item in sample_a)
                    total_char_len = char_len_l + char_len_a
                    regression_target = char_len_l / total_char_len if total_char_len > 0 else 0.5
                    
                    mixed_sentences = [item[0] for item in sample_l] + [item[0] for item in sample_a]
                    random.shuffle(mixed_sentences)
                    
                    flat_tokens = [token for sent in mixed_sentences for token in sent]
                    encoded = self.vocab.encode(flat_tokens)
                    
                    if len(encoded) > self.max_seq_len:
                        encoded = encoded[:self.max_seq_len]
                    else:
                        encoded = encoded + [0] * (self.max_seq_len - len(encoded))
                        
                self.static_samples.append((article_idx, encoded, regression_target))
                
    def set_epoch(self, epoch: int, total_epochs: int):
        self.current_epoch = epoch
        self.total_epochs = total_epochs
            
    def __len__(self):
        return len(self.static_samples)
        
    def __getitem__(self, idx: int):
        article_idx, static_encoded, static_target = self.static_samples[idx]
        
        if not self.is_train:
            return torch.tensor(static_encoded, dtype=torch.long), torch.tensor(static_target, dtype=torch.float)
            
        p_dynamic = self.current_epoch / max(1, self.total_epochs - 1)
        
        if random.random() < p_dynamic:
            ls_sents = self.ls_data[article_idx]
            as_sents = self.as_data[article_idx]
            num_leicht = len(ls_sents)
            num_alltag = len(as_sents)
            
            start_l, end_l = sorted([random.randint(0, num_leicht), random.randint(0, num_leicht)])
            sample_l = ls_sents[start_l:end_l]
            
            start_a, end_a = sorted([random.randint(0, num_alltag), random.randint(0, num_alltag)])
            sample_a = as_sents[start_a:end_a]
            
            if len(sample_l) == 0 and len(sample_a) == 0:
                regression_target = 0.5
                encoded = [0] * self.max_seq_len
            else:
                char_len_l = sum(item[1] for item in sample_l)
                char_len_a = sum(item[1] for item in sample_a)
                total_char_len = char_len_l + char_len_a
                regression_target = char_len_l / total_char_len if total_char_len > 0 else 0.5
                
                mixed_sentences = [item[0] for item in sample_l] + [item[0] for item in sample_a]
                random.shuffle(mixed_sentences)
                
                flat_tokens = [token for sent in mixed_sentences for token in sent]
                encoded = self.vocab.encode(flat_tokens)
                
                if len(encoded) > self.max_seq_len:
                    encoded = encoded[:self.max_seq_len]
                else:
                    encoded = encoded + [0] * (self.max_seq_len - len(encoded))
                    
            return torch.tensor(encoded, dtype=torch.long), torch.tensor(regression_target, dtype=torch.float)
        else:
            return torch.tensor(static_encoded, dtype=torch.long), torch.tensor(static_target, dtype=torch.float)


class BiLSTMRegressor(nn.Module):
    """BiLSTM Architecture with Sigmoid output for Simplicity Regression [0.0, 1.0]."""
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
        super(BiLSTMRegressor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        out = self.fc(self.dropout(hidden))
        return self.sigmoid(out)


def parse_args():
    parser = argparse.ArgumentParser(description="Data Scaling Experiment for MixUp Metric Model")
    parser.add_argument('--csv_path', default="data/analysis/corpus_master.csv", help="Path to corpus_master.csv")
    parser.add_argument('--min_sim', type=float, default=0.80, help="Min semantic similarity filter")
    parser.add_argument('--max_sim', type=float, default=0.98, help="Max semantic similarity filter")
    parser.add_argument('--max_seq_len', type=int, default=256, help="Max token sequence length")
    parser.add_argument('--embedding_dim', type=int, default=128, help="Embedding dimension")
    parser.add_argument('--hidden_dim', type=int, default=128, help="LSTM hidden dimension")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--epochs', type=int, default=40, help="Training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--patience', type=int, default=8, help="Early stopping patience")
    
    # Scaling Parameters
    parser.add_argument('--mixtures_per_pair', type=int, default=20, help="Number of synthetic mixtures per article pair")
    parser.add_argument('--train_fraction', type=float, default=1.0, help="Fraction of training article pairs to use [0.0 - 1.0]")
    parser.add_argument('--experiment_group', type=str, default="mixtures_scaling", choices=["mixtures_scaling", "pairs_scaling"], help="Group tag")
    parser.add_argument('--experiment_name', type=str, default="mixup_scale_m20_f100", help="Unique name for this experiment run")
    
    # Output Paths
    parser.add_argument('--output_dir', default="results/experiments/data_scaling", help="Directory to save artifacts")
    parser.add_argument('--vocab_save_path', default="data/vocabs/mixup_vocab.json", help="Path to vocabulary json")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=== Starting Data Scaling Experiment: %s ===", args.experiment_name)
    logger.info("Params: mixtures_per_pair=%d, train_fraction=%.2f, max_seq_len=%d",
                args.mixtures_per_pair, args.train_fraction, args.max_seq_len)
    
    # 1. Load and Filter Corpus
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Corpus file not found: {args.csv_path}")
        
    df = pd.read_csv(args.csv_path)
    mask = (df["semantic_similarity_8192"] >= args.min_sim) & (df["semantic_similarity_8192"] <= args.max_sim)
    df_filtered = df[mask].dropna(subset=["ls_text", "as_text"]).reset_index(drop=True)
    total_filtered_pairs = len(df_filtered)
    logger.info("Filtered corpus pairs: %d", total_filtered_pairs)
    
    # 2. Strict Train / Val / Test Split (Held-out Test set is fixed for all runs!)
    train_val_df, test_df = train_test_split(df_filtered, test_size=0.10, random_state=args.seed)
    train_full_df, val_df = train_test_split(train_val_df, test_size=0.1111, random_state=args.seed)
    
    # Apply Train Fraction if specified (< 1.0)
    if args.train_fraction < 1.0:
        num_train_selected = max(10, int(len(train_full_df) * args.train_fraction))
        train_df = train_full_df.sample(n=num_train_selected, random_state=args.seed).reset_index(drop=True)
    else:
        train_df = train_full_df.reset_index(drop=True)
        
    logger.info("Article pairs used: Train=%d (%.1f%% of train split), Val=%d, Test=%d",
                len(train_df), (len(train_df)/len(train_full_df))*100, len(val_df), len(test_df))
    
    # 3. Setup NLP Sentencizer & Vocabulary
    nlp = spacy.blank("de")
    nlp.add_pipe("sentencizer")
    
    # Vocab is built from the full training set so vocabulary size remains consistent
    if os.path.exists(args.vocab_save_path):
        logger.info("Loading existing vocabulary from %s", args.vocab_save_path)
        with open(args.vocab_save_path, "r", encoding="utf-8") as f:
            stoi = json.load(f)
        vocab = Vocab([], max_size=25000, min_freq=2)
        vocab.stoi = stoi
        vocab.itos = {v: k for k, v in stoi.items()}
    else:
        logger.info("Building vocabulary from training articles...")
        all_train_tokens = []
        for _, row in tqdm(train_full_df.iterrows(), total=len(train_full_df), desc="Tokenizing for Vocab"):
            for text in [str(row["ls_text"]), str(row["as_text"])]:
                doc = nlp(text)
                all_train_tokens.extend([t.text.lower() for t in doc if not t.is_space])
        vocab = Vocab(all_train_tokens, max_size=25000, min_freq=2)
        os.makedirs(os.path.dirname(args.vocab_save_path), exist_ok=True)
        with open(args.vocab_save_path, "w", encoding="utf-8") as f:
            json.dump(vocab.stoi, f, ensure_ascii=False, indent=2)
        logger.info("Saved vocabulary with %d tokens to %s", len(vocab), args.vocab_save_path)
        
    # 4. Build Datasets
    logger.info("Constructing PyTorch Datasets...")
    train_dataset = MixupPyTorchDataset(
        train_df, vocab, nlp,
        max_seq_len=args.max_seq_len,
        mixtures_per_pair=args.mixtures_per_pair,
        total_epochs=args.epochs,
        is_train=True,
        seed=args.seed
    )
    val_dataset = MixupPyTorchDataset(
        val_df, vocab, nlp,
        max_seq_len=args.max_seq_len,
        mixtures_per_pair=20,  # Fixed 20 mixtures for validation
        total_epochs=args.epochs,
        is_train=False,
        seed=args.seed
    )
    test_dataset = MixupPyTorchDataset(
        test_df, vocab, nlp,
        max_seq_len=args.max_seq_len,
        mixtures_per_pair=20,  # Fixed 20 mixtures for test
        total_epochs=args.epochs,
        is_train=False,
        seed=args.seed
    )
    
    total_train_samples = len(train_dataset)
    logger.info("Dataset samples: Train=%d, Val=%d, Test=%d",
                total_train_samples, len(val_dataset), len(test_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 5. Initialize Model & Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)
    
    model = BiLSTMRegressor(len(vocab), args.embedding_dim, args.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5)
    
    model_save_path = os.path.join(args.output_dir, f"{args.experiment_name}_model.pt")
    
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch, args.epochs)
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x).squeeze()
            if preds.ndim == 0:
                preds = preds.unsqueeze(0)
                
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x).squeeze()
                if preds.ndim == 0:
                    preds = preds.unsqueeze(0)
                    
                loss = criterion(preds, batch_y)
                val_loss += loss.item()
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(batch_y.cpu().numpy().tolist())
                
        epoch_train_loss = epoch_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_mae = mean_absolute_error(all_targets, all_preds)
        
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_mae"].append(epoch_val_mae)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break
                
        scheduler.step()
        
    training_duration = time.time() - start_time
    logger.info("Training completed in %.2f seconds. Best Epoch: %d (Val Loss: %.4f)",
                training_duration, best_epoch, best_val_loss)
    
    # 6. Evaluation on Held-Out Test Split
    logger.info("Evaluating best model on held-out Test-Split...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).squeeze()
            if preds.ndim == 0:
                preds = preds.unsqueeze(0)
            test_preds.extend(preds.cpu().numpy().tolist())
            test_targets.extend(batch_y.numpy().tolist())
            
    test_mse = float(mean_squared_error(test_targets, test_preds))
    test_mae = float(mean_absolute_error(test_targets, test_preds))
    test_r2 = float(r2_score(test_targets, test_preds))
    p_corr, _ = pearsonr(test_targets, test_preds)
    s_corr, _ = spearmanr(test_targets, test_preds)
    
    # Binary classification accuracy on pure/thresholded test samples (threshold 0.5)
    binary_acc = float(np.mean([(p >= 0.5) == (t >= 0.5) for p, t in zip(test_preds, test_targets)]))
    
    results = {
        "experiment_name": args.experiment_name,
        "experiment_group": args.experiment_group,
        "mixtures_per_pair": args.mixtures_per_pair,
        "train_fraction": args.train_fraction,
        "num_train_article_pairs": len(train_df),
        "total_train_samples": total_train_samples,
        "num_val_pairs": len(val_df),
        "num_test_pairs": len(test_df),
        "max_seq_len": args.max_seq_len,
        "best_epoch": best_epoch,
        "total_epochs_run": len(history["train_loss"]),
        "training_time_seconds": round(training_duration, 2),
        "best_val_loss": round(best_val_loss, 5),
        "test_mse": round(test_mse, 5),
        "test_mae": round(test_mae, 5),
        "test_r2": round(test_r2, 5),
        "test_pearson_r": round(float(p_corr), 5),
        "test_spearman_rho": round(float(s_corr), 5),
        "test_binary_acc": round(binary_acc * 100.0, 2),
        "model_save_path": model_save_path,
        "history": history
    }
    
    metrics_path = os.path.join(args.output_dir, f"{args.experiment_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    logger.info("=== Results for %s ===", args.experiment_name)
    logger.info("Test MSE: %.5f | Test MAE: %.5f | R2: %.4f | Pearson: %.4f | Acc: %.2f%%",
                test_mse, test_mae, test_r2, p_corr, binary_acc * 100.0)
    logger.info("Saved metrics to %s", metrics_path)

    # 7. Save Loss Curve Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        epochs_range = range(1, len(history["train_loss"]) + 1)
        plt.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
        plt.plot(epochs_range, history["val_loss"], label="Val Loss", marker="s")
        plt.title(f"Lernkurve: {args.experiment_name}")
        plt.xlabel("Epoche")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plot_path = os.path.join(args.output_dir, f"{args.experiment_name}_loss_curve.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved loss curve plot to %s", plot_path)
    except Exception as e:
        logger.warning("Could not generate plot for %s: %s", args.experiment_name, e)


if __name__ == "__main__":
    main()
