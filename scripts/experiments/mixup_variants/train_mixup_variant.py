#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Experiment: Train BiLSTM MixUp Regressor Variants
=============================================================================
Trains the four MixUp simplicity regression model variants:
  1. Variante A (Statisch):      Static mixing generated in dataset init (p_dynamic = 0.0)
  2. Variante B (Dynamisch):     Pure dynamic mixing per batch in __getitem__ (p_dynamic = 1.0)
  3. Variante C (Hybrid):        Linear dynamic schedule (p_dynamic = epoch / max_epochs)
  4. Variante D (Hybrid+Cyclic): Linear dynamic schedule + CosineAnnealingWarmRestarts LR

Saves checkpoints to results/models/mixup_variants/ and vocabs to data/mixup_variants/.
=============================================================================
"""

import os
import sys
import datetime
import random
import argparse
import json
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# LOGGING CLASS
# ==============================================================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ==============================================================================
# SEED CONFIGURATION
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# VOCABULARY
# ==============================================================================
class Vocab:
    def __init__(self, token_list=None, max_size=25000, min_freq=2, stoi_dict=None):
        if stoi_dict is not None:
            self.stoi = stoi_dict
            self.itos = {idx: token for token, idx in stoi_dict.items()}
        else:
            counter = Counter(token_list or [])
            self.itos = {0: "<pad>", 1: "<unk>"}
            self.stoi = {"<pad>": 0, "<unk>": 1}
            for token, freq in counter.most_common(max_size):
                if freq >= min_freq and token not in self.stoi:
                    idx = len(self.stoi)
                    self.stoi[token] = idx
                    self.itos[idx] = token

    def __len__(self):
        return len(self.stoi)

    def encode(self, tokens):
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]


# ==============================================================================
# PYTORCH DATASET
# ==============================================================================
class MixupPyTorchDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocab,
        nlp_sentencizer,
        max_seq_len: int = 150,
        mixtures_per_pair: int = 160,
        variant: str = "hybrid",
        is_train: bool = True,
        seed: int = 42
    ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.variant = variant
        self.is_train = is_train
        self.current_epoch = 0
        self.total_epochs = 100

        self.ls_data = []
        self.as_data = []
        self.static_samples = []

        set_seed(seed if is_train else 99)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Segmentiere ({'Train' if is_train else 'Val'})", leave=False):
            ls_doc = nlp_sentencizer(str(row.get("ls_text", "")))
            as_doc = nlp_sentencizer(str(row.get("as_text", "")))

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

        print(f"[{'Train' if is_train else 'Val'}] Generiert: {len(self.static_samples)} statische Samples ({len(self.ls_data)} Artikelpaare).")

    def set_epoch(self, epoch: int, total_epochs: int):
        self.current_epoch = epoch
        self.total_epochs = total_epochs

    def __len__(self):
        return len(self.static_samples)

    def __getitem__(self, idx):
        article_idx, static_encoded, static_target = self.static_samples[idx]

        if not self.is_train:
            return torch.tensor(static_encoded, dtype=torch.long), torch.tensor(static_target, dtype=torch.float)

        # Sampling-Wahrscheinlichkeit je nach Modellvariante bestimmen
        if self.variant == "static":
            p_dynamic = 0.0
        elif self.variant == "dynamic":
            p_dynamic = 1.0
        else:  # "hybrid" oder "hybrid_cyclic"
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
                encoded = [0] * self.max_seq_len
                regression_target = 0.5
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


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
class BiLSTMRegressor(nn.Module):
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
        return self.sigmoid(out).squeeze(-1)


# ==============================================================================
# MAIN TRAINING ROUTINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM MixUp Regressor Variants")
    parser.add_argument("--variant", choices=["static", "dynamic", "hybrid", "hybrid_cyclic"], default="hybrid_cyclic",
                        help="MixUp training strategy variant")
    parser.add_argument("--csv_path", default="data/analysis/corpus_master.csv", help="Path to corpus CSV")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="LSTM hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--max_sim", type=float, default=0.98, help="Max semantic similarity filter")
    parser.add_argument("--min_sim", type=float, default=0.80, help="Min semantic similarity filter")
    parser.add_argument("--max_seq_len", type=int, default=150, help="Max token sequence length")
    parser.add_argument("--mixtures_per_pair", type=int, default=160, help="Mixtures per article pair in training")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--model_save_path", default=None, help="Output checkpoint path")
    parser.add_argument("--vocab_save_path", default="data/mixup_variants/mixup_vocab.json", help="Vocab JSON path")
    parser.add_argument("--log_dir", default="results/logs/experiments/mixup_variants", help="Log directory")
    parser.add_argument("--plot_dir", default="results/plots/experiments/mixup_variants", help="Plot directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.model_save_path is None:
        args.model_save_path = f"results/models/mixup_variants/bilstm_mixup_regression_{args.variant}.pt"

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.model_save_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.vocab_save_path)), exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"train_mixup_{args.variant}_{timestamp}.log")
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout

    print("=============================================================================")
    print(f"MixUp Regressor Training - Variante: {args.variant}")
    print(f"Log-Datei: {log_file}")
    print(f"Model-Ziel: {args.model_save_path}")
    print(f"Vocab-Ziel: {args.vocab_save_path}")
    print("=============================================================================")

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Nutze Device: {device}")

    # Fallback für CSV Pfad
    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        fallback = "data/analysis/information_loss_analysis_cleaned.csv"
        if os.path.exists(fallback):
            print(f"Hinweis: {csv_path} nicht gefunden, nutze Fallback {fallback}")
            csv_path = fallback

    df = pd.read_csv(csv_path)
    sim_col = "semantic_similarity_8192" if "semantic_similarity_8192" in df.columns else "semantic_similarity"
    mask = (df[sim_col] >= args.min_sim) & (df[sim_col] <= args.max_sim)
    df_filtered = df[mask].dropna(subset=["ls_text", "as_text"])
    print(f"Gefilterte Artikelpaare (Sim in [{args.min_sim}, {args.max_sim}]): {len(df_filtered)}")

    nlp = spacy.blank("de")
    nlp.add_pipe("sentencizer")

    # Train / Val / Test Split
    train_val_df, test_df = train_test_split(df_filtered, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1111, random_state=42)
    print(f"Splits -> Training: {len(train_df)} | Validierung: {len(val_df)} | Test: {len(test_df)}")

    # Vocabular laden oder erstellen
    if os.path.exists(args.vocab_save_path):
        print(f"Lade bestehendes Vokabular aus {args.vocab_save_path}...")
        with open(args.vocab_save_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
            stoi_dict = vocab_dict.get("stoi", vocab_dict)
        vocab = Vocab(stoi_dict=stoi_dict)
    else:
        print("Erstelle neues Vokabular aus Trainingsdaten...")
        all_train_tokens = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Vokab-Tokens sammeln"):
            for text in [str(row["ls_text"]), str(row["as_text"])]:
                doc = nlp(text)
                for token in doc:
                    if not token.is_space:
                        all_train_tokens.append(token.text.lower())
        vocab = Vocab(all_train_tokens, max_size=25000, min_freq=2)
        with open(args.vocab_save_path, "w", encoding="utf-8") as f:
            json.dump(vocab.stoi, f, ensure_ascii=False, indent=2)
        print(f"Vokabular gespeichert unter: {args.vocab_save_path}")

    print(f"Vokabulargröße: {len(vocab)} Tokens")

    # Datasets & DataLoader
    print("Initialisiere Datasets...")
    train_dataset = MixupPyTorchDataset(
        train_df, vocab, nlp, max_seq_len=args.max_seq_len, mixtures_per_pair=args.mixtures_per_pair,
        variant=args.variant, is_train=True, seed=args.seed
    )
    val_dataset = MixupPyTorchDataset(
        val_df, vocab, nlp, max_seq_len=args.max_seq_len, mixtures_per_pair=min(20, args.mixtures_per_pair),
        variant=args.variant, is_train=False, seed=args.seed
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Modell initialisieren
    model = BiLSTMRegressor(
        vocab_size=len(vocab),
        embed_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    if args.variant == "hybrid_cyclic":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5)
    else:
        scheduler = None

    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    best_val_loss = float("inf")
    counter = 0

    print("\n--- Starte Trainingsschleife ---")
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch, args.epochs)
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs}", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            if preds.ndim == 0:
                preds = preds.unsqueeze(0)

            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validierung
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
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

        print(f"Epoch {epoch+1:02d} | Train MSE: {epoch_train_loss:.4f} | Val MSE: {epoch_val_loss:.4f} | Val MAE: {epoch_val_mae:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), args.model_save_path)
            print(f"  => Neuer Bestwert! Modell gespeichert unter {args.model_save_path}")
            counter = 0
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early Stopping ausgelöst nach {epoch+1} Epochen (Patience: {args.patience}).")
                break

        if scheduler is not None:
            scheduler.step()

    # Trainingsverlauf plotten
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss (MSE)", color="#e74c3c")
    plt.plot(history["val_loss"], label="Val Loss (MSE)", color="#3498db")
    plt.title(f"Trainingsverlauf - MixUp Variante: {args.variant}", fontsize=13, weight="bold")
    plt.xlabel("Epoche")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    loss_plot_path = os.path.join(args.plot_dir, f"train_loss_{args.variant}.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Lernkurve gespeichert unter: {loss_plot_path}")

    # Zusammenfassungsmetriken speichern
    summary_path = os.path.splitext(args.model_save_path)[0] + "_metrics.json"
    summary_data = {
        "variant": args.variant,
        "best_val_loss": float(best_val_loss),
        "min_val_mae": float(min(history["val_mae"])),
        "epochs_trained": len(history["train_loss"]),
        "history": history
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Metriken gespeichert unter: {summary_path}")
    print(f"=== Training von {args.variant} erfolgreich abgeschlossen ===")


if __name__ == "__main__":
    main()
