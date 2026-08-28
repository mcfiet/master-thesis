#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Similarity Threshold Experiment: MixUp Simplicity Regressor Training & Eval
=============================================================================
Untersucht den Einfluss des Ähnlichkeits-Schwellenwerts s_min in {0.60, 0.70, 0.80}
(bei s_max = 0.98) auf das Training und die Generalisierung des BiLSTM MixUp Regressors.

Nutzung der exakten Hyperparameter aus run_pipeline:
- max_seq_len = 1024
- mixtures_per_pair = 160
- batch_size = 64
- embedding_dim = 128, hidden_dim = 128
- lr = 0.001
- epochs = 80, patience = 15
- CosineAnnealingWarmRestarts (T_0 = 10, T_mult = 1, eta_min = 1e-5)
- Hybrid Dataloader (statisch -> dynamisch)

Evaluation:
1. In-Domain Held-Out Testset (10% Split): MSE, MAE, R², Pearson r, Spearman rho, Acc.
2. Out-of-Domain Lebenshilfe Dataset (data/lebenshilfe/lebenshilfe_dataset_clean.json):
   OOD MAE, OOD MSE, Separation ROC-AUC (AS vs. LS), Perfect Pair Match Rate.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("MixUpSimilarityExperiment")


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
    def __init__(self, tokens: List[str] = None, max_size: int = 25000, min_freq: int = 2):
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = {0: "<pad>", 1: "<unk>"}
        
        if tokens:
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
        max_seq_len: int = 1024,
        mixtures_per_pair: int = 160,
        total_epochs: int = 80,
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
    parser = argparse.ArgumentParser(description="Similarity Threshold Experiment: MixUp Regressor")
    parser.add_argument('--csv_path', default="data/analysis/corpus_master.csv", help="Path to master corpus")
    parser.add_argument('--lh_dataset_path', default="data/lebenshilfe/lebenshilfe_dataset_clean.json", help="Path to Lebenshilfe dataset")
    parser.add_argument('--min_sim', type=float, default=0.80, help="Min cosine similarity threshold (e.g. 0.60, 0.70, 0.80)")
    parser.add_argument('--max_sim', type=float, default=0.98, help="Max cosine similarity threshold")
    parser.add_argument('--max_seq_len', type=int, default=1024, help="Max sequence length in tokens")
    parser.add_argument('--mixtures_per_pair', type=int, default=160, help="Number of MixUp mixtures per pair")
    parser.add_argument('--embedding_dim', type=int, default=128, help="Embedding dimension")
    parser.add_argument('--hidden_dim', type=int, default=128, help="LSTM hidden dimension")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--epochs', type=int, default=80, help="Training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--patience', type=int, default=15, help="Early stopping patience")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--output_dir', default="results/experiments/similarity_threshold", help="Output directory")
    parser.add_argument('--experiment_name', type=str, default=None, help="Custom experiment name")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    if args.experiment_name is None:
        sim_str = f"{int(round(args.min_sim * 100)):02d}"
        args.experiment_name = f"mixup_sim_{sim_str}"
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"=== Starting MixUp Similarity Threshold Experiment: {args.experiment_name} ===")
    logger.info(f"Similarity Filter: [{args.min_sim:.2f}, {args.max_sim:.2f}] | SeqLen: {args.max_seq_len} | Mixtures: {args.mixtures_per_pair}")
    
    # 1. Daten laden und filtern
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Corpus file not found: {args.csv_path}")
        
    df_raw = pd.read_csv(args.csv_path)
    mask = (df_raw["semantic_similarity_8192"] >= args.min_sim) & (df_raw["semantic_similarity_8192"] <= args.max_sim)
    df_filtered = df_raw[mask].dropna(subset=["ls_text", "as_text"]).reset_index(drop=True)
    total_filtered_pairs = len(df_filtered)
    logger.info(f"Verfügbare Artikelpaare im Schwellenwert [{args.min_sim:.2f}, {args.max_sim:.2f}]: {total_filtered_pairs} (aus {len(df_raw)} gesamt)")
    
    # 2. Fester Split (80% Train, 10% Val, 10% Test)
    train_val_df, test_df = train_test_split(df_filtered, test_size=0.10, random_state=args.seed)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1111, random_state=args.seed)
    
    logger.info(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Held-Out Test={len(test_df)}")
    
    # 3. NLP Pipeline & Vokabular
    nlp = spacy.blank("de")
    nlp.add_pipe("sentencizer")
    
    logger.info("Erstelle Vokabular aus Trainingsartikeln...")
    all_train_tokens = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Tokenisiere Trainingsdaten", leave=False):
        for text in [str(row["ls_text"]), str(row["as_text"])]:
            doc = nlp(text)
            all_train_tokens.extend([t.text.lower() for t in doc if not t.is_space])
            
    vocab = Vocab(all_train_tokens, max_size=25000, min_freq=2)
    logger.info(f"Vokabulargröße: {len(vocab)} Tokens")
    
    vocab_save_path = os.path.join(args.output_dir, f"{args.experiment_name}_vocab.json")
    with open(vocab_save_path, "w", encoding="utf-8") as f:
        json.dump(vocab.stoi, f, ensure_ascii=False, indent=2)
        
    # 4. Datasets
    logger.info("Konstruiere PyTorch Datasets...")
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
        mixtures_per_pair=20,  # Feste 20 Mischungen für Validierung
        total_epochs=args.epochs,
        is_train=False,
        seed=args.seed
    )
    test_dataset = MixupPyTorchDataset(
        test_df, vocab, nlp,
        max_seq_len=args.max_seq_len,
        mixtures_per_pair=20,  # Feste 20 Mischungen für Test
        total_epochs=args.epochs,
        is_train=False,
        seed=args.seed
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 5. Model & Training
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    logger.info(f"Trainiere Modell auf Device: {device}")
    
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
            preds = model(batch_x).squeeze(-1)
            if preds.ndim == 0:
                preds = preds.unsqueeze(0)
                
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x).squeeze(-1)
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
                logger.info(f"Early Stopping ausgelöst in Epoche {epoch + 1}.")
                break
                
        scheduler.step()
        
    train_duration = time.time() - start_time
    logger.info(f"Training beendet nach {train_duration:.2f}s. Bester Val Loss: {best_val_loss:.4f} (Epoche {best_epoch})")
    
    # 6. In-Domain Evaluation (Held-out Test Split)
    logger.info("Starte In-Domain Testset Evaluation...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).squeeze(-1)
            if preds.ndim == 0:
                preds = preds.unsqueeze(0)
            test_preds.extend(preds.cpu().numpy().tolist())
            test_targets.extend(batch_y.numpy().tolist())
            
    test_mse = float(mean_squared_error(test_targets, test_preds))
    test_mae = float(mean_absolute_error(test_targets, test_preds))
    test_r2 = float(r2_score(test_targets, test_preds))
    p_corr, _ = pearsonr(test_targets, test_preds)
    s_corr, _ = spearmanr(test_targets, test_preds)
    test_bin_acc = float(np.mean([(p >= 0.5) == (t >= 0.5) for p, t in zip(test_preds, test_targets)])) * 100.0
    
    # 7. Out-of-Domain Evaluation (Lebenshilfe Benchmark)
    logger.info("Starte Out-of-Domain Evaluation auf dem Lebenshilfe-Datensatz...")
    ood_results = {}
    if os.path.exists(args.lh_dataset_path):
        with open(args.lh_dataset_path, "r", encoding="utf-8") as f:
            lh_data = json.load(f)
            
        as_texts = [item.get("as_text") or item.get("source_text", "") for item in lh_data]
        ls_texts = [item.get("ls_text") or item.get("target_text", "") for item in lh_data]
        
        def score_texts(texts: List[str]) -> List[float]:
            scores = []
            for t in texts:
                doc = nlp(str(t))
                toks = [tok.text.lower() for tok in doc if not tok.is_space]
                encoded = vocab.encode(toks)[:args.max_seq_len]
                padded = encoded + [0] * max(0, args.max_seq_len - len(encoded))
                tensor = torch.tensor([padded], dtype=torch.long, device=device)
                with torch.no_grad():
                    pred = float(model(tensor).squeeze().cpu().item())
                scores.append(pred)
            return scores
            
        as_scores = score_texts(as_texts)
        ls_scores = score_texts(ls_texts)
        
        # OOD Metriken
        lh_y_true = [0.0] * len(as_scores) + [1.0] * len(ls_scores)
        lh_y_pred = as_scores + ls_scores
        
        ood_mae = float(mean_absolute_error(lh_y_true, lh_y_pred))
        ood_mse = float(mean_squared_error(lh_y_true, lh_y_pred))
        
        # ROC-AUC (Trennschärfe zwischen AS und LS)
        ood_auc = float(roc_auc_score([0] * len(as_scores) + [1] * len(ls_scores), lh_y_pred))
        
        # Perfect Pair Match Rate (LS Score > AS Score für paralleles Dokumentenpaar)
        pair_correct = sum([ls_s > as_s for ls_s, as_s in zip(ls_scores, as_scores)])
        perfect_pair_rate = (pair_correct / len(lh_data)) * 100.0
        
        ood_results = {
            "ood_num_pairs": len(lh_data),
            "ood_mae": round(ood_mae, 4),
            "ood_mse": round(ood_mse, 4),
            "ood_separation_auc": round(ood_auc, 4),
            "ood_perfect_pair_match_pct": round(perfect_pair_rate, 2),
            "ood_mean_score_as": round(float(np.mean(as_scores)), 4),
            "ood_mean_score_ls": round(float(np.mean(ls_scores)), 4),
            "ood_score_delta": round(float(np.mean(ls_scores) - np.mean(as_scores)), 4)
        }
        
        logger.info(f"OOD Lebenshilfe -> AUC: {ood_auc:.4f} | MAE: {ood_mae:.4f} | Perfect Match: {perfect_pair_rate:.1f}% | Δ(LS-AS): {ood_results['ood_score_delta']:.4f}")
    else:
        logger.warning(f"Lebenshilfe Datensatz unter {args.lh_dataset_path} nicht gefunden!")
        
    # Zusammenfassung
    metrics_summary = {
        "experiment_name": args.experiment_name,
        "model_type": "MixUp Regressor",
        "min_sim": float(args.min_sim),
        "max_sim": float(args.max_sim),
        "num_total_filtered_pairs": total_filtered_pairs,
        "num_train_pairs": len(train_df),
        "num_val_pairs": len(val_df),
        "num_test_pairs": len(test_df),
        "vocab_size": len(vocab),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 5),
        "training_time_seconds": round(train_duration, 2),
        "in_domain_test_mse": round(test_mse, 5),
        "in_domain_test_mae": round(test_mae, 5),
        "in_domain_test_r2": round(test_r2, 4),
        "in_domain_pearson_r": round(float(p_corr), 4),
        "in_domain_spearman_rho": round(float(s_corr), 4),
        "in_domain_binary_acc_pct": round(test_bin_acc, 2),
        **ood_results,
        "model_save_path": model_save_path
    }
    
    # JSON Summary speichern
    json_path = os.path.join(args.output_dir, f"{args.experiment_name}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
        
    # Loss Plot speichern
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        epochs_range = range(1, len(history["train_loss"]) + 1)
        plt.plot(epochs_range, history["train_loss"], label="Train Loss (MSE)", marker="o")
        plt.plot(epochs_range, history["val_loss"], label="Val Loss (MSE)", marker="s")
        plt.title(f"Lernkurve: {args.experiment_name} (Sim [{args.min_sim:.2f}, {args.max_sim:.2f}])")
        plt.xlabel("Epoche")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plot_path = os.path.join(args.output_dir, f"{args.experiment_name}_loss_curve.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"Konnte Plot nicht speichern: {e}")
        
    logger.info(f"[ERFOLG] MixUp Experiment {args.experiment_name} abgeschlossen!")
    logger.info(f"Gespeichert in: {json_path}")


if __name__ == "__main__":
    main()
