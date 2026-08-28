#!/usr/bin/env python3
"""
scripts/experiments/evaluate_classifier_length_experiment.py

Vergleicht alle Klassifikationsansätze über Sequenzlängen und Hierarchieebenen:
1. Satz-Klassifikator (Satzebene N=7.146)
2. Satz-Klassifikator (Majority Vote / Ratio r_LS, N=74 Dokumente)
3. Artikel-Klassifikator (256 Tokens, N=74 Dokumente)
4. Artikel-Klassifikator (512 Tokens, N=74 Dokumente)
5. Artikel-Klassifikator (1024 Tokens, N=74 Dokumente)
6. Baselines: Flesch & Wiener Sachtextformel

Speichert CSVs und generiert hochauflösende Grafiken in results/plots/experiments/classifier_length/.
"""

import os
import sys
import json
import argparse
import ast
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import spacy
import textstat
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr, wasserstein_distance, ks_2samp
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, precision_recall_curve, average_precision_score,
    confusion_matrix
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
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


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        v = json.load(f)
    return v["stoi"] if "stoi" in v else v


def evaluate_classifier_metrics(y_true: np.ndarray, y_scores: np.ndarray, name: str, level: str, threshold: float = 0.5) -> Dict[str, Any]:
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
    parser.add_argument("--sent_model_path", default="results/models/bilstm_sentence_classifier.pt")
    parser.add_argument("--sent_vocab_path", default="data/vocabs/sentence_vocab.json")
    parser.add_argument("--art_256_model", default="results/models/classifier_length_exp/bilstm_article_classifier_256.pt")
    parser.add_argument("--art_256_vocab", default="data/classifier_length_exp/article_vocab_256.json")
    parser.add_argument("--art_512_model", default="results/models/classifier_length_exp/bilstm_article_classifier_512.pt")
    parser.add_argument("--art_512_vocab", default="data/classifier_length_exp/article_vocab_512.json")
    parser.add_argument("--art_1024_model", default="results/models/classifier_length_exp/bilstm_article_classifier_1024.pt")
    parser.add_argument("--art_1024_vocab", default="data/classifier_length_exp/article_vocab_1024.json")
    parser.add_argument("--output_csv", default="results/evaluation/classifier_length_comparison_eval.csv")
    parser.add_argument("--summary_json", default="results/evaluation/classifier_length_summary.json")
    parser.add_argument("--plot_dir", default="results/plots/experiments/classifier_length")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    nlp = spacy.blank("de")
    nlp.add_pipe("sentencizer")

    # Fallback for 1024 if not yet in classifier_length_exp
    if not os.path.exists(args.art_1024_model) and os.path.exists("results/models/bilstm_article_classifier.pt"):
        args.art_1024_model = "results/models/bilstm_article_classifier.pt"
    if not os.path.exists(args.art_1024_vocab) and os.path.exists("data/vocabs/article_vocab.json"):
        args.art_1024_vocab = "data/vocabs/article_vocab.json"

    # Modelle laden
    models = {}
    vocabs = {}

    def load_clf(m_path, v_path, key):
        if os.path.exists(m_path) and os.path.exists(v_path):
            vocabs[key] = load_vocab(v_path)
            st = torch.load(m_path, map_location=device)
            if "model_state_dict" in st: st = st["model_state_dict"]
            emb_weight = st.get("embedding.weight", None)
            v_size = emb_weight.shape[0] if emb_weight is not None else len(vocabs[key])
            m = BiLSTMClassifier(v_size).to(device)
            m.load_state_dict(st)
            m.eval()
            models[key] = m
            print(f"-> Geladen: {key} ({m_path})")

    load_clf(args.sent_model_path, args.sent_vocab_path, "sent")
    load_clf(args.art_256_model, args.art_256_vocab, "art_256")
    load_clf(args.art_512_model, args.art_512_vocab, "art_512")
    load_clf(args.art_1024_model, args.art_1024_vocab, "art_1024")

    # Lebenshilfe Dataset laden
    with open(args.lh_dataset_path, "r", encoding="utf-8") as f:
        lh_data = json.load(f)
    print(f"Lebenshilfe Artikelpaare: {len(lh_data)}")

    records = []
    all_ls_sents_preds, all_as_sents_preds = [], []

    for item in tqdm(lh_data, desc="Klassifikator-Inferenz"):
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

        # 1. Satz-Klassifikator
        if "sent" in models:
            def score_sents(doc):
                probs = []
                for s in doc.sents:
                    toks = [t.text.lower() for t in s if not t.is_space]
                    if toks:
                        enc = [vocabs["sent"].get(t, 1) for t in toks][:100] + [0] * (100 - len(toks))
                        with torch.no_grad():
                            p = torch.sigmoid(models["sent"](torch.tensor([enc]).to(device)).squeeze()).item()
                        probs.append(p)
                return probs

            ls_sp = score_sents(ls_doc)
            as_sp = score_sents(as_doc)
            all_ls_sents_preds.extend(ls_sp)
            all_as_sents_preds.extend(as_sp)

            row["ls_sent_ratio"] = float(np.mean(np.array(ls_sp) >= 0.5)) if ls_sp else 0.5
            row["as_sent_ratio"] = float(np.mean(np.array(as_sp) >= 0.5)) if as_sp else 0.5
            row["ls_sent_prob_mean"] = float(np.mean(ls_sp)) if ls_sp else 0.5
            row["as_sent_prob_mean"] = float(np.mean(as_sp)) if as_sp else 0.5
            row["pair_match_sent_ratio"] = row["ls_sent_ratio"] > row["as_sent_ratio"]
            row["pair_match_sent_mv"] = (row["ls_sent_ratio"] >= 0.5) and (row["as_sent_ratio"] < 0.5)

        # 2. Artikel-Klassifikatoren (256, 512, 1024)
        for length_key, max_l in [("art_256", 256), ("art_512", 512), ("art_1024", 1024)]:
            if length_key in models:
                ls_toks = [t.text.lower() for t in ls_doc if not t.is_space][:max_l]
                as_toks = [t.text.lower() for t in as_doc if not t.is_space][:max_l]
                enc_l = [vocabs[length_key].get(t, 1) for t in ls_toks] + [0] * (max_l - len(ls_toks))
                enc_a = [vocabs[length_key].get(t, 1) for t in as_toks] + [0] * (max_l - len(as_toks))
                with torch.no_grad():
                    p_ls = torch.sigmoid(models[length_key](torch.tensor([enc_l]).to(device)).squeeze()).item()
                    p_as = torch.sigmoid(models[length_key](torch.tensor([enc_a]).to(device)).squeeze()).item()
                row[f"ls_{length_key}_prob"] = p_ls
                row[f"as_{length_key}_prob"] = p_as
                row[f"pair_match_{length_key}"] = p_ls > p_as

        records.append(row)

    df_eval = pd.DataFrame(records)
    df_eval.to_csv(args.output_csv, index=False)
    print(f"Detail-CSVs gespeichert unter: {args.output_csv}")

    # Master Tabelle berechnen
    y_true_doc = np.array([1] * len(df_eval) + [0] * len(df_eval))
    bench_records = []

    # 1. Satzebene (N=7146)
    if all_ls_sents_preds and all_as_sents_preds:
        y_true_s = np.array([1] * len(all_ls_sents_preds) + [0] * len(all_as_sents_preds))
        scores_s = np.array(all_ls_sents_preds + all_as_sents_preds)
        m_s = evaluate_classifier_metrics(y_true_s, scores_s, "BiLSTM Satz-Klassifikator (Sentence-Level)", f"Satzebene ($N={len(y_true_s)}$)", 0.5)
        m_s["Perfect Pair Match"] = "N/A (Satzebene)"
        bench_records.append(m_s)

    # 2. Majority Vote / Ratio
    if "ls_sent_ratio" in df_eval.columns:
        sc = np.concatenate([df_eval["ls_sent_ratio"], df_eval["as_sent_ratio"]])
        m = evaluate_classifier_metrics(y_true_doc, sc, "BiLSTM Satz-Klassifikator (Majority Vote)", "Dokument ($r_{LS}$ Ratio)", 0.5)
        m["Perfect Pair Match"] = f"{df_eval['pair_match_sent_ratio'].mean()*100:.1f}% ({df_eval['pair_match_sent_ratio'].sum()}/{len(df_eval)})"
        bench_records.append(m)

    # 3. Artikel 256
    if "ls_art_256_prob" in df_eval.columns:
        sc = np.concatenate([df_eval["ls_art_256_prob"], df_eval["as_art_256_prob"]])
        m = evaluate_classifier_metrics(y_true_doc, sc, "BiLSTM Artikel-Klassifikator (256)", "Dokument ($256$ Tokens)", 0.5)
        m["Perfect Pair Match"] = f"{df_eval['pair_match_art_256'].mean()*100:.1f}% ({df_eval['pair_match_art_256'].sum()}/{len(df_eval)})"
        bench_records.append(m)

    # 4. Artikel 512
    if "ls_art_512_prob" in df_eval.columns:
        sc = np.concatenate([df_eval["ls_art_512_prob"], df_eval["as_art_512_prob"]])
        m = evaluate_classifier_metrics(y_true_doc, sc, "BiLSTM Artikel-Klassifikator (512)", "Dokument ($512$ Tokens)", 0.5)
        m["Perfect Pair Match"] = f"{df_eval['pair_match_art_512'].mean()*100:.1f}% ({df_eval['pair_match_art_512'].sum()}/{len(df_eval)})"
        bench_records.append(m)

    # 5. Artikel 1024
    if "ls_art_1024_prob" in df_eval.columns:
        sc = np.concatenate([df_eval["ls_art_1024_prob"], df_eval["as_art_1024_prob"]])
        m = evaluate_classifier_metrics(y_true_doc, sc, "BiLSTM Artikel-Klassifikator (1024)", "Dokument ($1.024$ Tokens)", 0.5)
        m["Perfect Pair Match"] = f"{df_eval['pair_match_art_1024'].mean()*100:.1f}% ({df_eval['pair_match_art_1024'].sum()}/{len(df_eval)})"
        bench_records.append(m)

    # 6. Baselines
    sc_f = np.concatenate([df_eval["ls_flesch"], df_eval["as_flesch"]])
    m_f = evaluate_classifier_metrics(y_true_doc, (sc_f >= 55.0).astype(int), "Flesch Reading Ease (Baseline)", "Dokument (Silben/Satz)", 0.5)
    m_f["Perfect Pair Match"] = f"{df_eval['pair_match_flesch'].mean()*100:.1f}% ({df_eval['pair_match_flesch'].sum()}/{len(df_eval)})"
    bench_records.append(m_f)

    sc_w = np.concatenate([-df_eval["ls_wiener"], -df_eval["as_wiener"]])
    m_w = evaluate_classifier_metrics(y_true_doc, (sc_w >= -6.5).astype(int), "Wiener Sachtextformel (Baseline)", "Dokument (Wortlänge)", 0.5)
    m_w["Perfect Pair Match"] = f"{df_eval['pair_match_wiener'].mean()*100:.1f}% ({df_eval['pair_match_wiener'].sum()}/{len(df_eval)})"
    bench_records.append(m_w)

    df_master = pd.DataFrame(bench_records)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(bench_records, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("KLASSIFIKATOR BENCHMARK TABELLE (LEBENSHILFE GOLDSTANDARD)")
    print("=" * 80)
    cols = ["Modell", "Ebene / Input", "Ø Score (LS)", "Ø Score (AS)", "Separation (Δ)", "Balanced Acc", "ROC-AUC", "Perfect Pair Match"]
    print(df_master[cols].to_string(index=False))

    # PLOTS ERZEUGEN
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    # Plot 1: KDE Verteilung nach Längen
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if "ls_sent_ratio" in df_eval.columns:
        sns.kdeplot(df_eval["ls_sent_ratio"], ax=axes[0, 0], color="#2ca02c", fill=True, label="LS")
        sns.kdeplot(df_eval["as_sent_ratio"], ax=axes[0, 0], color="#1f77b4", fill=True, label="AS")
        axes[0, 0].set_title("(a) Satz-Klassifikator ($r_{LS}$ Ratio)", fontweight="bold")
        axes[0, 0].legend()

    for idx, (k, title) in enumerate([("art_256", "(b) Artikel-Klassifikator (256 Tokens)"),
                                       ("art_512", "(c) Artikel-Klassifikator (512 Tokens)"),
                                       ("art_1024", "(d) Artikel-Klassifikator (1024 Tokens)")]):
        ax = axes[(idx + 1) // 2, (idx + 1) % 2]
        if f"ls_{k}_prob" in df_eval.columns:
            sns.kdeplot(df_eval[f"ls_{k}_prob"], ax=ax, color="#2ca02c", fill=True, label="LS")
            sns.kdeplot(df_eval[f"as_{k}_prob"], ax=ax, color="#1f77b4", fill=True, label="AS")
            ax.set_title(title, fontweight="bold")
            ax.legend()

    plt.tight_layout()
    p1 = os.path.join(args.plot_dir, "classifier_kde_length_comparison.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Plot: {p1}")

    # Plot 2: ROC & PR Kurven
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    curves = []
    if "ls_sent_ratio" in df_eval.columns:
        curves.append(("Satz-Klassifikator (Ratio)", np.concatenate([df_eval["ls_sent_ratio"], df_eval["as_sent_ratio"]]), "#1f77b4", "-"))
    if "ls_art_256_prob" in df_eval.columns:
        curves.append(("Artikel-Clf (256)", np.concatenate([df_eval["ls_art_256_prob"], df_eval["as_art_256_prob"]]), "#2ca02c", "--"))
    if "ls_art_512_prob" in df_eval.columns:
        curves.append(("Artikel-Clf (512)", np.concatenate([df_eval["ls_art_512_prob"], df_eval["as_art_512_prob"]]), "#ff7f0e", "-."))
    if "ls_art_1024_prob" in df_eval.columns:
        curves.append(("Artikel-Clf (1024)", np.concatenate([df_eval["ls_art_1024_prob"], df_eval["as_art_1024_prob"]]), "#d62728", ":"))

    for name, sc, col, ls in curves:
        fpr, tpr, _ = roc_curve(y_true_doc, sc)
        auc_v = roc_auc_score(y_true_doc, sc)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc_v:.4f})", color=col, linestyle=ls, lw=2)

        prec, rec, _ = precision_recall_curve(y_true_doc, sc)
        ap_v = average_precision_score(y_true_doc, sc)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap_v:.4f})", color=col, linestyle=ls, lw=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("ROC-Kurven (Klassifikator Längenvergleich)", fontweight="bold")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend(loc="lower right")

    axes[1].set_title("Precision-Recall-Kurven", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    p2 = os.path.join(args.plot_dir, "classifier_roc_pr_length_comparison.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Plot: {p2}")

    # Plot 3: Confusion Matrices
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    cm_data = []
    if all_ls_sents_preds and all_as_sents_preds:
        cm_data.append(("Satzebene ($N=7.146$)", (np.array(all_ls_sents_preds + all_as_sents_preds) >= 0.5).astype(int), np.array([1]*len(all_ls_sents_preds) + [0]*len(all_as_sents_preds))))
    if "ls_sent_ratio" in df_eval.columns:
        cm_data.append((r"Satz-Ratio ($r_{LS} \geq 0.5$)", (np.concatenate([df_eval["ls_sent_ratio"], df_eval["as_sent_ratio"]]) >= 0.5).astype(int), y_true_doc))
    if "ls_art_512_prob" in df_eval.columns:
        cm_data.append(("Artikel (512)", (np.concatenate([df_eval["ls_art_512_prob"], df_eval["as_art_512_prob"]]) >= 0.5).astype(int), y_true_doc))
    if "ls_art_1024_prob" in df_eval.columns:
        cm_data.append(("Artikel (1024)", (np.concatenate([df_eval["ls_art_1024_prob"], df_eval["as_art_1024_prob"]]) >= 0.5).astype(int), y_true_doc))

    for idx, (title, y_p, y_t) in enumerate(cm_data):
        cm = confusion_matrix(y_t, y_p)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx], xticklabels=["AS (0)", "LS (1)"], yticklabels=["AS (0)", "LS (1)"])
        axes[idx].set_title(title, fontweight="bold")
        axes[idx].set_xlabel("Vorhergesagt")
        axes[idx].set_ylabel("Tatsächlich")

    plt.tight_layout()
    p3 = os.path.join(args.plot_dir, "classifier_confusion_matrices.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Plot: {p3}")

    print("\n[ERFOLG] Klassifikator-Längen-Evaluation abgeschlossen.")


if __name__ == "__main__":
    main()
