#!/usr/bin/env python3
"""
scripts/experiments/evaluate_regressor_length_experiment.py

Vergleicht alle Regressionsansätze über Sequenzlängen und Hierarchieebenen:
1. Satz-Regressor (Token-Weighted Mean)
2. Satz-Regressor (Arithmetic Mean)
3. Satz-Regressor (Median & P20 Quantil)
4. MixUp-Regressor (256 Tokens)
5. MixUp-Regressor (512 Tokens)
6. MixUp-Regressor (1024 Tokens)
7. Baselines: Flesch & Wiener Sachtextformel

Speichert CSVs und generiert hochauflösende Grafiken in results/plots/experiments/regressor_length/.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import spacy
import textstat
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr, wasserstein_distance, ks_2samp
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, precision_recall_curve, average_precision_score
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class BiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
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
        out = self.fc(self.dropout(hidden))
        return self.sigmoid(out).squeeze(-1)


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        v = json.load(f)
    return v["stoi"] if "stoi" in v else v


def evaluate_regressor_metrics(y_true: np.ndarray, y_scores: np.ndarray, name: str, level: str, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_scores >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    auc = float(roc_auc_score(y_true, y_scores))
    ap = float(average_precision_score(y_true, y_scores))

    ls_scores = y_scores[y_true == 1]
    as_scores = y_scores[y_true == 0]
    mean_ls, std_ls = float(np.mean(ls_scores)), float(np.std(ls_scores))
    mean_as, std_as = float(np.mean(as_scores)), float(np.std(as_scores))
    separation = float(mean_ls - mean_as)
    w_dist = float(wasserstein_distance(ls_scores, as_scores))
    ks_stat = float(ks_2samp(ls_scores, as_scores).statistic)

    return {
        "Modell": name,
        "Ebene / Input": level,
        "Ø Score (LS)": f"{mean_ls:.3f} ± {std_ls:.2f}",
        "Ø Score (AS)": f"{mean_as:.3f} ± {std_as:.2f}",
        "Separation (Δ)": f"{separation:.3f}",
        "Balanced Acc": f"{bacc * 100:.2f}%",
        "Recall (LS)": f"{rec * 100:.2f}%",
        "F1-Score": f"{f1 * 100:.2f}%",
        "ROC-AUC": f"{auc:.4f}",
        "PR-AUC (AP)": f"{ap:.4f}",
        "Wasserstein (W1)": f"{w_dist:.4f}",
        "KS-Statistik (D)": f"{ks_stat:.4f}",
        "_bacc": bacc,
        "_auc": auc,
        "_w_dist": w_dist,
        "_separation": separation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lh_dataset_path", default="data/lebenshilfe/lebenshilfe_dataset_clean.json")
    parser.add_argument("--mix_256_model", default="results/models/regressor_length_exp/bilstm_mixup_regression_256.pt")
    parser.add_argument("--mix_256_vocab", default="data/regressor_length_exp/mixup_vocab_256.json")
    parser.add_argument("--mix_512_model", default="results/models/regressor_length_exp/bilstm_mixup_regression_512.pt")
    parser.add_argument("--mix_512_vocab", default="data/regressor_length_exp/mixup_vocab_512.json")
    parser.add_argument("--mix_1024_model", default="results/models/regressor_length_exp/bilstm_mixup_regression_1024.pt")
    parser.add_argument("--mix_1024_vocab", default="data/regressor_length_exp/mixup_vocab_1024.json")
    parser.add_argument("--output_csv", default="results/evaluation/regressor_length_comparison_eval.csv")
    parser.add_argument("--summary_json", default="results/evaluation/regressor_length_summary.json")
    parser.add_argument("--plot_dir", default="results/plots/experiments/regressor_length")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    nlp = spacy.blank("de")
    nlp.add_pipe("sentencizer")

    # Fallbacks für 1024 falls im Root gespeichert
    if not os.path.exists(args.mix_1024_model) and os.path.exists("results/models/bilstm_mixup_regression.pt"):
        args.mix_1024_model = "results/models/bilstm_mixup_regression.pt"
    if not os.path.exists(args.mix_1024_vocab) and os.path.exists("data/vocabs/mixup_vocab.json"):
        args.mix_1024_vocab = "data/vocabs/mixup_vocab.json"

    models = {}
    vocabs = {}

    def load_reg(m_path, v_path, key):
        if os.path.exists(m_path) and os.path.exists(v_path):
            vocabs[key] = load_vocab(v_path)
            st = torch.load(m_path, map_location=device)
            if "model_state_dict" in st: st = st["model_state_dict"]
            emb_w = st.get("embedding.weight", None)
            v_size = emb_w.shape[0] if emb_w is not None else len(vocabs[key])
            m = BiLSTMRegressor(v_size).to(device)
            m.load_state_dict(st)
            m.eval()
            models[key] = m
            print(f"-> Geladen: {key} ({m_path})")

    load_reg(args.mix_256_model, args.mix_256_vocab, "mix_256")
    load_reg(args.mix_512_model, args.mix_512_vocab, "mix_512")
    load_reg(args.mix_1024_model, args.mix_1024_vocab, "mix_1024")

    # Lebenshilfe Dataset laden
    with open(args.lh_dataset_path, "r", encoding="utf-8") as f:
        lh_data = json.load(f)
    print(f"Lebenshilfe Artikelpaare: {len(lh_data)}")

    records = []
    for item in tqdm(lh_data, desc="Regressor-Inferenz"):
        pair_id = item.get("pair_id", 0)
        ls_fn = item.get("ls_filename", f"pair_{pair_id}_ls")
        as_fn = item.get("as_filename", f"pair_{pair_id}_as")
        ls_text = str(item.get("ls_text", ""))
        as_text = str(item.get("as_text", ""))

        ls_doc = nlp(ls_text)
        as_doc = nlp(as_text)

        row = {
            "pair_id": pair_id,
            "ls_filename": ls_fn,
            "as_filename": as_fn,
            "ls_flesch": textstat.flesch_reading_ease(ls_text),
            "as_flesch": textstat.flesch_reading_ease(as_text),
            "ls_wiener": textstat.wiener_sachtextformel(ls_text, 1),
            "as_wiener": textstat.wiener_sachtextformel(as_text, 1),
        }
        row["pair_match_flesch"] = row["ls_flesch"] > row["as_flesch"]
        row["pair_match_wiener"] = row["ls_wiener"] < row["as_wiener"]

        # Dokument-Regressoren (256, 512, 1024)
        for length_key, max_l in [("mix_256", 256), ("mix_512", 512), ("mix_1024", 1024)]:
            if length_key in models:
                ls_toks = [t.text.lower() for t in ls_doc if not t.is_space][:max_l]
                as_toks = [t.text.lower() for t in as_doc if not t.is_space][:max_l]
                enc_l = [vocabs[length_key].get(t, 1) for t in ls_toks] or [0]
                enc_a = [vocabs[length_key].get(t, 1) for t in as_toks] or [0]
                with torch.no_grad():
                    p_ls = models[length_key](torch.tensor([enc_l]).to(device)).item()
                    p_as = models[length_key](torch.tensor([enc_a]).to(device)).item()
                row[f"ls_{length_key}_score"] = p_ls
                row[f"as_{length_key}_score"] = p_as
                row[f"pair_match_{length_key}"] = p_ls > p_as

        records.append(row)

    df_eval = pd.DataFrame(records)
    df_eval.to_csv(args.output_csv, index=False)
    print(f"Detail-CSVs gespeichert unter: {args.output_csv}")

    y_true_doc = np.array([1] * len(df_eval) + [0] * len(df_eval))
    bench_records = []

    # 1. MixUp Regressor 256
    if "ls_mix_256_score" in df_eval.columns:
        sc = np.concatenate([df_eval["ls_mix_256_score"], df_eval["as_mix_256_score"]])
        m = evaluate_regressor_metrics(y_true_doc, sc, "BiLSTM MixUp Regressor (256)", "Dokument ($256$ Tokens)", 0.5)
        m["Perfect Pair Match"] = f"{df_eval['pair_match_mix_256'].mean()*100:.1f}% ({df_eval['pair_match_mix_256'].sum()}/{len(df_eval)})"
        bench_records.append(m)

    # 2. MixUp Regressor 512
    if "ls_mix_512_score" in df_eval.columns:
        sc = np.concatenate([df_eval["ls_mix_512_score"], df_eval["as_mix_512_score"]])
        m = evaluate_regressor_metrics(y_true_doc, sc, "BiLSTM MixUp Regressor (512)", "Dokument ($512$ Tokens)", 0.5)
        m["Perfect Pair Match"] = f"{df_eval['pair_match_mix_512'].mean()*100:.1f}% ({df_eval['pair_match_mix_512'].sum()}/{len(df_eval)})"
        bench_records.append(m)

    # 3. MixUp Regressor 1024
    if "ls_mix_1024_score" in df_eval.columns:
        sc = np.concatenate([df_eval["ls_mix_1024_score"], df_eval["as_mix_1024_score"]])
        m = evaluate_regressor_metrics(y_true_doc, sc, "BiLSTM MixUp Regressor (1024)", "Dokument ($1.024$ Tokens)", 0.5)
        m["Perfect Pair Match"] = f"{df_eval['pair_match_mix_1024'].mean()*100:.1f}% ({df_eval['pair_match_mix_1024'].sum()}/{len(df_eval)})"
        bench_records.append(m)

    # 4. Baselines
    sc_f = np.concatenate([df_eval["ls_flesch"], df_eval["as_flesch"]])
    m_f = evaluate_regressor_metrics(y_true_doc, (sc_f >= 55.0).astype(int), "Flesch Reading Ease (Baseline)", "Dokument (Silben/Satz)", 0.5)
    m_f["Perfect Pair Match"] = f"{df_eval['pair_match_flesch'].mean()*100:.1f}% ({df_eval['pair_match_flesch'].sum()}/{len(df_eval)})"
    bench_records.append(m_f)

    sc_w = np.concatenate([-df_eval["ls_wiener"], -df_eval["as_wiener"]])
    m_w = evaluate_regressor_metrics(y_true_doc, (sc_w >= -6.5).astype(int), "Wiener Sachtextformel (Baseline)", "Dokument (Wortlänge)", 0.5)
    m_w["Perfect Pair Match"] = f"{df_eval['pair_match_wiener'].mean()*100:.1f}% ({df_eval['pair_match_wiener'].sum()}/{len(df_eval)})"
    bench_records.append(m_w)

    df_master = pd.DataFrame(bench_records)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(bench_records, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("REGRESSOR LÄNGEN-BENCHMARK TABELLE (LEBENSHILFE GOLDSTANDARD)")
    print("=" * 80)
    cols = ["Modell", "Ebene / Input", "Ø Score (LS)", "Ø Score (AS)", "Separation (Δ)", "Balanced Acc", "ROC-AUC", "Perfect Pair Match"]
    print(df_master[cols].to_string(index=False))

    # PLOTS ERZEUGEN
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    # Plot 1: KDE Dichten der 3 Längen
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (k, title) in enumerate([("mix_256", "(a) MixUp Regressor (256 Tokens)"),
                                       ("mix_512", "(b) MixUp Regressor (512 Tokens)"),
                                       ("mix_1024", "(c) MixUp Regressor (1024 Tokens)")]):
        ax = axes[idx]
        if f"ls_{k}_score" in df_eval.columns:
            sns.kdeplot(df_eval[f"ls_{k}_score"], ax=ax, color="#2ca02c", fill=True, label="Leichte Sprache (LS)")
            sns.kdeplot(df_eval[f"as_{k}_score"], ax=ax, color="#1f77b4", fill=True, label="Alltagssprache (AS)")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Simplicity Score (0 = AS, 1 = LS)")
            ax.legend(loc="upper center")

    plt.suptitle("KDE-Dichteverteilungen der MixUp-Regressoren nach Kontextlänge", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p1 = os.path.join(args.plot_dir, "regressor_kde_length_comparison.png")
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot: {p1}")

    # Plot 2: ROC & PR Kurven
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    curves = []
    if "ls_mix_256_score" in df_eval.columns:
        curves.append(("MixUp (256)", np.concatenate([df_eval["ls_mix_256_score"], df_eval["as_mix_256_score"]]), "#1f77b4", "--"))
    if "ls_mix_512_score" in df_eval.columns:
        curves.append(("MixUp (512)", np.concatenate([df_eval["ls_mix_512_score"], df_eval["as_mix_512_score"]]), "#ff7f0e", "-."))
    if "ls_mix_1024_score" in df_eval.columns:
        curves.append(("MixUp (1024)", np.concatenate([df_eval["ls_mix_1024_score"], df_eval["as_mix_1024_score"]]), "#d62728", "-"))

    for name, sc, col, ls in curves:
        fpr, tpr, _ = roc_curve(y_true_doc, sc)
        auc_v = roc_auc_score(y_true_doc, sc)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc_v:.4f})", color=col, linestyle=ls, lw=2)

        prec, rec, _ = precision_recall_curve(y_true_doc, sc)
        ap_v = average_precision_score(y_true_doc, sc)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap_v:.4f})", color=col, linestyle=ls, lw=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("ROC-Kurven (Regressor Längenvergleich)", fontweight="bold")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend(loc="lower right")

    axes[1].set_title("Precision-Recall-Kurven", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    p2 = os.path.join(args.plot_dir, "regressor_roc_pr_length_comparison.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Plot: {p2}")

    # Plot 3: Paarweise Trajektorien
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x_pairs = np.arange(1, len(df_eval) + 1)
    if "ls_sent_reg_wmean" in df_eval.columns:
        axes[0].plot(x_pairs, df_eval["ls_sent_reg_wmean"], "o-", color="#2ca02c", label="LS", lw=1.8, markersize=5)
        axes[0].plot(x_pairs, df_eval["as_sent_reg_wmean"], "s-", color="#1f77b4", label="AS", lw=1.8, markersize=5)
        axes[0].fill_between(x_pairs, df_eval["ls_sent_reg_wmean"], df_eval["as_sent_reg_wmean"], color="#2ca02c", alpha=0.15)
        axes[0].set_title("(a) BiLSTM Satz-Regressor (Token-Weighted)", fontweight="bold")
        axes[0].set_xlabel("Paar ID")
        axes[0].set_ylabel("Simplicity Score")
        axes[0].legend()

    if "ls_mix_1024_score" in df_eval.columns:
        axes[1].plot(x_pairs, df_eval["ls_mix_1024_score"], "o-", color="#2ca02c", label="LS", lw=1.8, markersize=5)
        axes[1].plot(x_pairs, df_eval["as_mix_1024_score"], "s-", color="#1f77b4", label="AS", lw=1.8, markersize=5)
        axes[1].fill_between(x_pairs, df_eval["ls_mix_1024_score"], df_eval["as_mix_1024_score"], color="#2ca02c", alpha=0.15)
        axes[1].set_title("(b) BiLSTM MixUp Regressor (1024 Tokens)", fontweight="bold")
        axes[1].set_xlabel("Paar ID")
        axes[1].set_ylabel("Simplicity Score")
        axes[1].legend()

    plt.tight_layout()
    p3 = os.path.join(args.plot_dir, "regressor_pairwise_trajectories.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Plot: {p3}")

    print("\n[ERFOLG] Regressor-Längen-Evaluation abgeschlossen.")


if __name__ == "__main__":
    main()
