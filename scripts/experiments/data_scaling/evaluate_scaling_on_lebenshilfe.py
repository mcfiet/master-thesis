#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Zero-Shot Evaluation on Lebenshilfe Gold-Standard Dataset for All Scaling Runs
=============================================================================
Evaluates all trained BiLSTM MixUp Scaling checkpoints on the external
clean Lebenshilfe test dataset (data/lebenshilfe/lebenshilfe_dataset_clean.json).

Computes:
  - Mean predicted score on Everyday German (AS)
  - Mean predicted score on Leichte Sprache (LS)
  - Separation Delta (LS - AS)
  - Pairwise Ranking Accuracy (LS > AS)
  - Binary Threshold Accuracy (>= 0.5)
  - ROC-AUC Score separating AS and LS
  - MAE to Gold-Standard targets (0.0 for AS, 1.0 for LS)

Outputs:
  - results/experiments/data_scaling/scaling_lh_summary.csv
  - results/experiments/data_scaling/plots/lh_scaling_auc_delta.png
  - results/experiments/data_scaling/plots/lh_kde_separation_m80.png
=============================================================================
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, mean_absolute_error, roc_curve
import spacy


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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Data Scaling models on Lebenshilfe dataset")
    parser.add_argument("--lh_dataset", default="data/lebenshilfe/lebenshilfe_dataset_clean.json", help="Path to LH dataset")
    parser.add_argument("--vocab_path", default="data/data_scaling/mixup_vocab.json", help="Path to vocabulary")
    parser.add_argument("--models_dir", default="results/experiments/data_scaling", help="Directory containing .pt models")
    parser.add_argument("--output_csv", default="results/experiments/data_scaling/scaling_lh_summary.csv", help="Output summary CSV")
    parser.add_argument("--plots_dir", default="results/experiments/data_scaling/plots", help="Directory to save plots")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Max token sequence length")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    if not os.path.exists(args.lh_dataset):
        print(f"Lebenshilfe dataset not found at: {args.lh_dataset}")
        return

    # Load Spacy
    nlp = spacy.blank("de")

    # Load Vocab
    vocab_path = args.vocab_path
    if not os.path.exists(vocab_path) and os.path.exists("data/vocabs/mixup_vocab.json"):
        print(f"Notice: {vocab_path} not found, falling back to data/vocabs/mixup_vocab.json")
        vocab_path = "data/vocabs/mixup_vocab.json"

    with open(vocab_path, "r", encoding="utf-8") as f:
        full_stoi = json.load(f)

    # Load LH Data
    with open(args.lh_dataset, "r", encoding="utf-8") as f:
        lh_data = json.load(f)

    as_texts = [d.get("as_text", "") for d in lh_data]
    ls_texts = [d.get("ls_text", "") for d in lh_data]

    model_files = sorted(glob.glob(os.path.join(args.models_dir, "*_model.pt")))
    if not model_files:
        print(f"No model checkpoint files (*_model.pt) found in {args.models_dir}")
        return

    # Determine checkpoint vocab size from the first checkpoint
    first_state = torch.load(model_files[0], map_location="cpu", weights_only=False)
    state_dict = first_state["model_state_dict"] if "model_state_dict" in first_state else first_state
    vocab_size = state_dict["embedding.weight"].shape[0]

    stoi = {k: v for k, v in full_stoi.items() if v < vocab_size}

    def tokenize(texts: list, max_len: int = 256) -> torch.Tensor:
        tokenized = []
        for text in texts:
            doc = nlp(str(text or ""))
            tokens = [t.text.lower() for t in doc if not t.is_space]
            ids = [stoi.get(t, stoi.get("<unk>", 1)) for t in tokens][:max_len]
            if len(ids) == 0:
                ids = [0]
            tokenized.append(ids)
        padded = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, seq in enumerate(tokenized):
            padded[i, :len(seq)] = seq
        return torch.tensor(padded, dtype=torch.long)

    x_as = tokenize(as_texts, max_len=args.max_seq_len).to(args.device)
    x_ls = tokenize(ls_texts, max_len=args.max_seq_len).to(args.device)

    records = []
    all_predictions = {}

    for mf in model_files:
        exp_name = os.path.basename(mf).replace("_model.pt", "")
        model = BiLSTMRegressor(vocab_size)
        state = torch.load(mf, map_location=args.device, weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.to(args.device)
        model.eval()

        with torch.no_grad():
            preds_as = model(x_as).cpu().numpy()
            preds_ls = model(x_ls).cpu().numpy()

        all_predictions[exp_name] = {"as": preds_as, "ls": preds_ls}

        mean_as = float(np.mean(preds_as))
        mean_ls = float(np.mean(preds_ls))
        delta = float(mean_ls - mean_as)
        pairwise_acc = float(np.mean(preds_ls > preds_as) * 100.0)

        y_true = np.array([0.0] * len(preds_as) + [1.0] * len(preds_ls))
        y_pred = np.concatenate([preds_as, preds_ls])

        mae = float(mean_absolute_error(y_true, y_pred))
        auc = float(roc_auc_score(y_true, y_pred) * 100.0)
        bin_acc = float(np.mean([(p >= 0.5) == (t >= 0.5) for p, t in zip(y_pred, y_true)]) * 100.0)

        group = "mixtures_scaling" if "mixtures" in exp_name else "pairs_scaling"

        # Extract scaling parameter
        if group == "mixtures_scaling":
            m_val = int(exp_name.split("_m")[-1]) if "_m" in exp_name else 20
            f_val = 1.0
        else:
            f_str = exp_name.split("_f")[-1] if "_f" in exp_name else "100"
            if f_str == "100":
                f_val = 1.0
            else:
                f_val = float(int(f_str)) / 100.0 if len(f_str) == 3 else float(f"0.{f_str}")
            m_val = 20

        records.append({
            "experiment_name": exp_name,
            "experiment_group": group,
            "mixtures_per_pair": m_val,
            "train_fraction": f_val,
            "lh_mean_as": round(mean_as, 4),
            "lh_mean_ls": round(mean_ls, 4),
            "lh_delta": round(delta, 4),
            "lh_pairwise_acc": round(pairwise_acc, 2),
            "lh_binary_acc": round(bin_acc, 2),
            "lh_auc": round(auc, 2),
            "lh_mae": round(mae, 4)
        })

    df = pd.DataFrame(records).sort_values(by=["experiment_group", "mixtures_per_pair", "train_fraction"])
    df.to_csv(args.output_csv, index=False)
    print(f"Successfully saved Lebenshilfe evaluation results to: {args.output_csv}")

    # Also merge into scaling_summary.csv if available
    summary_path = os.path.join(args.models_dir, "scaling_summary.csv")
    if os.path.exists(summary_path):
        try:
            sum_df = pd.read_csv(summary_path)
            merged = pd.merge(sum_df, df[["experiment_name", "lh_mean_as", "lh_mean_ls", "lh_delta", "lh_pairwise_acc", "lh_binary_acc", "lh_auc", "lh_mae"]], on="experiment_name", how="left")
            merged.to_csv(summary_path, index=False)
            print(f"Merged Lebenshilfe metrics into master summary: {summary_path}")
        except Exception as e:
            print(f"Could not merge into scaling_summary.csv: {e}")

    # Print Summary Table
    print("\n" + "=" * 95)
    print("LEBENSHILFE (LH) ZERO-SHOT BENCHMARK EVALUATION ACROSS ALL SCALING STEPS")
    print("=" * 95)
    print(df[["experiment_name", "experiment_group", "lh_mean_as", "lh_mean_ls", "lh_delta", "lh_pairwise_acc", "lh_binary_acc", "lh_auc", "lh_mae"]].to_string(index=False))
    print("=" * 95)

    # -------------------------------------------------------------
    # PLOT 1: Zero-Shot Separation Delta & ROC-AUC Scaling Curves
    # -------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot A: Mixtures Scaling
    m_df = df[df["experiment_group"] == "mixtures_scaling"].sort_values("mixtures_per_pair")
    if not m_df.empty:
        ax1.plot(m_df["mixtures_per_pair"], m_df["lh_auc"], marker="o", color="#1f77b4", linewidth=2.2, label="ROC-AUC [%]")
        ax1.set_xlabel(r"MixUp-Multiplikator ($M = \mathrm{mixtures\_per\_pair}$)")
        ax1.set_ylabel("Zero-Shot ROC-AUC [%]", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1_twin = ax1.twinx()
        ax1_twin.plot(m_df["mixtures_per_pair"], m_df["lh_delta"], marker="s", linestyle="--", color="#ff7f0e", linewidth=2.2, label=r"Separation $\Delta$")
        ax1_twin.set_ylabel(r"Separations-Abstand $\Delta = \overline{LS} - \overline{AS}$", color="#ff7f0e")
        ax1_twin.tick_params(axis="y", labelcolor="#ff7f0e")
        ax1.set_title("A: Lebenshilfe-Separation über MixUp ($M$)", pad=12, fontweight="bold")

    # Subplot B: Pairs Scaling
    p_df = df[df["experiment_group"] == "pairs_scaling"].sort_values("train_fraction")
    if not p_df.empty:
        ax2.plot(p_df["train_fraction"] * 100, p_df["lh_auc"], marker="o", color="#2ca02c", linewidth=2.2, label="ROC-AUC [%]")
        ax2.set_xlabel(r"Trainingsfraktion $F$ [%] ($M=20$)")
        ax2.set_ylabel("Zero-Shot ROC-AUC [%]", color="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")
        ax2_twin = ax2.twinx()
        ax2_twin.plot(p_df["train_fraction"] * 100, p_df["lh_delta"], marker="s", linestyle="--", color="#d62728", linewidth=2.2, label=r"Separation $\Delta$")
        ax2_twin.set_ylabel(r"Separations-Abstand $\Delta = \overline{LS} - \overline{AS}$", color="#d62728")
        ax2_twin.tick_params(axis="y", labelcolor="#d62728")
        ax2.set_title("B: Lebenshilfe-Separation über Basispaare ($F$)", pad=12, fontweight="bold")

    fig.tight_layout()
    plot_scaling_path = os.path.join(args.plots_dir, "lh_scaling_auc_delta.png")
    plt.savefig(plot_scaling_path, dpi=300)
    plt.close()
    print(f"Saved LH scaling curve plot to: {plot_scaling_path}")

    # -------------------------------------------------------------
    # PLOT 2: Best Model (M=80) KDE Distribution & ROC Curve
    # -------------------------------------------------------------
    best_exp = "scale_mixtures_m80" if "scale_mixtures_m80" in all_predictions else model_files[-1]
    if best_exp in all_predictions:
        preds_as = all_predictions[best_exp]["as"]
        preds_ls = all_predictions[best_exp]["ls"]

        fig, (ax_kde, ax_roc) = plt.subplots(1, 2, figsize=(14, 5))

        # KDE
        sns.kdeplot(preds_as, ax=ax_kde, color="#d62728", fill=True, alpha=0.4, linewidth=2, label=fr"Alltagssprache (AS, $\mu$={np.mean(preds_as):.2f})")
        sns.kdeplot(preds_ls, ax=ax_kde, color="#2ca02c", fill=True, alpha=0.4, linewidth=2, label=fr"Leichte Sprache (LS, $\mu$={np.mean(preds_ls):.2f})")
        ax_kde.axvline(x=0.5, color="black", linestyle=":", label="Entscheidungsschwelle 0.5")
        ax_kde.set_title(f"Lebenshilfe Dichteverteilung ({best_exp})", pad=12, fontweight="bold")
        ax_kde.set_xlabel(r"Vorhergesagter Einfachheits-Score $\hat{y} \in [0, 1]$")
        ax_kde.set_ylabel("Dichte (KDE)")
        ax_kde.legend(loc="upper center")

        # ROC Curve
        y_true = np.array([0.0] * len(preds_as) + [1.0] * len(preds_ls))
        y_pred = np.concatenate([preds_as, preds_ls])
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_val = roc_auc_score(y_true, y_pred) * 100.0

        ax_roc.plot(fpr, tpr, color="#1f77b4", linewidth=2.5, label=f"ROC-Kurve (AUC = {auc_val:.2f}%)")
        ax_roc.plot([0, 1], [0, 1], color="grey", linestyle="--")
        ax_roc.set_title(f"Lebenshilfe Zero-Shot ROC-Kurve ({best_exp})", pad=12, fontweight="bold")
        ax_roc.set_xlabel("False Positive Rate (FPR)")
        ax_roc.set_ylabel("True Positive Rate (TPR)")
        ax_roc.legend(loc="lower right")

        fig.tight_layout()
        plot_kde_path = os.path.join(args.plots_dir, "lh_kde_separation_m80.png")
        plt.savefig(plot_kde_path, dpi=300)
        plt.close()
        print(f"Saved LH best model KDE plot to: {plot_kde_path}")


if __name__ == "__main__":
    main()
