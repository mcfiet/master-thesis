#!/usr/bin/env python3
"""
scripts/evaluation/evaluate_merlin.py

Umfassende Evaluation ALLER Modellparadigmen auf dem standardisierten
MERLIN CEFR-Benchmark (Boyd et al., 2014 / UniversalCEFR):
1.033 deutsche Fließtexte (ganze Prüfungsaufsätze/Briefe) mit offiziellen CEFR-Niveaus (A1, A2, B1, B2, C1, C2).

1. Regressoren:
   - BiLSTM MixUp Regressoren (256, 512, 1024 Tokens)
2. Klassifikatoren:
   - BiLSTM Satz-Klassifikator (P(LS))
   - BiLSTM Artikel-Klassifikatoren (256, 512, 1024 Tokens)
3. Traditionelle Baselines:
   - Flesch Reading Ease
   - Wiener Sachtextformel
   - LIX Lesbarkeitsindex
"""

import os
import sys
import json
import argparse
import urllib.request
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import spacy
import textstat
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
        return self.fc(self.dropout(hidden)).squeeze(-1)


def load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        v = json.load(f)
    return v["stoi"] if "stoi" in v else v


def evaluate_series(preds: np.ndarray, y_simp: np.ndarray, y_comp: np.ndarray, name: str, cat: str) -> Dict[str, Any]:
    # Pearson & Spearman mit CEFR Simplicity (1.0 = A1, 0.0 = C2)
    r_simp, _ = pearsonr(preds, y_simp)
    rho_simp, _ = spearmanr(preds, y_simp)
    tau_simp, _ = kendalltau(preds, y_simp)

    # Korrelation mit numerischem CEFR-Level (A1=1, A2=2, B1=3, B2=4, C1=5, C2=6)
    r_comp, _ = pearsonr(preds, -y_comp)

    # MSE & MAE auf normalisierter Skala
    preds_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
    rmse = float(np.sqrt(mean_squared_error(y_simp, preds_norm)))
    mae = float(mean_absolute_error(y_simp, preds_norm))

    return {
        "Modell": name,
        "Kategorie": cat,
        "Pearson r (Simplicity)": float(r_simp) if not np.isnan(r_simp) else 0.0,
        "Spearman ρ (Rang)": float(rho_simp) if not np.isnan(rho_simp) else 0.0,
        "Kendall τ": float(tau_simp) if not np.isnan(tau_simp) else 0.0,
        "Pearson r (Complexity)": float(r_comp) if not np.isnan(r_comp) else 0.0,
        "RMSE (normiert)": float(rmse),
        "MAE (normiert)": float(mae),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_json", default="data/analysis/merlin/merlin_de.json")
    parser.add_argument("--benchmark_csv", default="data/analysis/merlin/merlin_texts.csv")
    parser.add_argument("--output_csv", default="results/evaluation/merlin_all_models_eval.csv")
    parser.add_argument("--summary_json", default="results/evaluation/merlin_summary.json")
    parser.add_argument("--plot_dir", default="results/plots/experiments/merlin")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.benchmark_csv), exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    nlp = spacy.blank("de")

    # 1. Benchmark-Daten laden / herunterladen falls nicht vorhanden
    if not os.path.exists(args.benchmark_csv):
        print("Lade MERLIN Dataset von UniversalCEFR herunter...")
        url = "https://huggingface.co/datasets/UniversalCEFR/merlin_de/raw/main/merlin-de.json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        with open(args.benchmark_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        df_raw = pd.DataFrame(data)
        cefr_order = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
        df_raw["cefr_num"] = df_raw["cefr_level"].map(cefr_order)
        df_raw["cefr_simplicity"] = 1.0 - ((df_raw["cefr_num"] - 1.0) / 5.0)
        df_raw["word_count"] = df_raw["text"].apply(lambda s: len(str(s).split()))
        df_raw.to_csv(args.benchmark_csv, index=False)
        print(f"MERLIN Dataset gespeichert: {args.benchmark_csv} ({len(df_raw)} Dokumente)")

    df = pd.read_csv(args.benchmark_csv)
    texts = df["text"].tolist()
    y_simp = df["cefr_simplicity"].to_numpy()
    y_comp = df["cefr_num"].to_numpy()

    df_out = df.copy()

    # 2. Baselines berechnen
    print("Berechne traditionelle Lesbarkeitsbaselines auf Dokumentebene...")
    flesch_scores = np.array([textstat.flesch_reading_ease(str(t)) for t in texts])
    wiener_scores = np.array([-textstat.wiener_sachtextformel(str(t), 1) for t in texts])
    lix_scores = np.array([-textstat.lix(str(t)) for t in texts])

    df_out["Pred_Flesch"] = flesch_scores
    df_out["Pred_Wiener"] = wiener_scores
    df_out["Pred_LIX"] = lix_scores

    summary_records = []
    summary_records.append(evaluate_series(flesch_scores, y_simp, y_comp, "Flesch Reading Ease", "Baseline"))
    summary_records.append(evaluate_series(wiener_scores, y_simp, y_comp, "Wiener Sachtextformel", "Baseline"))
    summary_records.append(evaluate_series(lix_scores, y_simp, y_comp, "LIX Lesbarkeitsindex", "Baseline"))

    # 3. Vorab-Tokenisierung aller Dokumente (einmalig für maximale Performance)
    print("Tokenisiere alle Dokumente einmalig...")
    docs_tokens = []
    for t in texts:
        doc = nlp(str(t or ""))
        docs_tokens.append([tok.text.lower() for tok in doc if not tok.is_space])

    # 4. Neuronale Modelle definieren
    models_config = [
        ("BiLSTM MixUp Regressor (256)", "results/models/regressor_length_exp/bilstm_mixup_regression_256.pt", "data/regressor_length_exp/mixup_vocab_256.json", "reg", 256, "Regressor (256)"),
        ("BiLSTM MixUp Regressor (512)", "results/models/regressor_length_exp/bilstm_mixup_regression_512.pt", "data/regressor_length_exp/mixup_vocab_512.json", "reg", 512, "Regressor (512)"),
        ("BiLSTM MixUp Regressor (1024)", "results/models/bilstm_mixup_regression.pt", "data/vocabs/mixup_vocab.json", "reg", 1024, "Regressor (1024)"),
        ("BiLSTM Satz-Klassifikator", "results/models/bilstm_sentence_classifier.pt", "data/vocabs/sentence_vocab.json", "clf", 100, "Klassifikator (Satz)"),
        ("BiLSTM Artikel-Klassifikator (256)", "results/models/classifier_length_exp/bilstm_article_classifier_256.pt", "data/classifier_length_exp/article_vocab_256.json", "clf", 256, "Klassifikator (256)"),
        ("BiLSTM Artikel-Klassifikator (512)", "results/models/classifier_length_exp/bilstm_article_classifier_512.pt", "data/classifier_length_exp/article_vocab_512.json", "clf", 512, "Klassifikator (512)"),
        ("BiLSTM Artikel-Klassifikator (1024)", "results/models/bilstm_article_classifier.pt", "data/vocabs/article_vocab.json", "clf", 1024, "Klassifikator (1024)"),
    ]

    for name, m_path, v_path, m_type, max_len, cat in models_config:
        if not os.path.exists(m_path):
            alt_m = m_path.replace("regressor_length_exp", "token_length_exp").replace("classifier_length_exp", "token_length_exp")
            if os.path.exists(alt_m): m_path = alt_m
        if not os.path.exists(v_path):
            alt_v = v_path.replace("regressor_length_exp", "token_length_exp").replace("classifier_length_exp", "token_length_exp")
            if os.path.exists(alt_v): v_path = alt_v

        if not os.path.exists(m_path) or not os.path.exists(v_path):
            print(f"[SKIP] {name}: Checkpoint oder Vokabular nicht gefunden ({m_path})")
            continue

        print(f"Evaluiere {name}...")
        vocab = load_vocab(v_path)
        st = torch.load(m_path, map_location=device)
        if "model_state_dict" in st: st = st["model_state_dict"]
        emb_w = st.get("embedding.weight", None)
        v_size = emb_w.shape[0] if emb_w is not None else len(vocab)

        if m_type == "reg":
            model = BiLSTMRegressor(v_size).to(device)
        else:
            model = BiLSTMClassifier(v_size).to(device)
        model.load_state_dict(st)
        model.eval()

        preds_list = []
        with torch.no_grad():
            for toks in docs_tokens:
                enc = [vocab.get(tok, 1) if vocab.get(tok, 1) < v_size else 1 for tok in toks][:max_len]
                if not enc:
                    enc = [0]
                inp = torch.tensor([enc], dtype=torch.long, device=device)
                if m_type == "reg":
                    p = model(inp).squeeze().item()
                else:
                    p = torch.sigmoid(model(inp)).squeeze().item()
                preds_list.append(p)
        preds = np.array(preds_list)

        df_out[f"Pred_{name.replace(' ', '_')}"] = preds
        summary_records.append(evaluate_series(preds, y_simp, y_comp, name, cat))

    df_out.to_csv(args.output_csv, index=False)
    print(f"Detail-Vorhersagen gespeichert: {args.output_csv}")

    df_summary = pd.DataFrame(summary_records)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("MERLIN CEFR BENCHMARK MASTER TABELLE (N=1.033 DEUTSCHE DOKUMENTE)")
    print("=" * 90)
    cols_show = ["Modell", "Kategorie", "Pearson r (Simplicity)", "Spearman ρ (Rang)", "Kendall τ", "MAE (normiert)"]
    print(df_summary[cols_show].to_string(index=False))

    # PLOTS GENERIEREN
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150

    # 1. Plot: Barchart der Pearson & Spearman Korrelationen
    plt.figure(figsize=(14, 6))
    df_plot = df_summary.sort_values(by="Pearson r (Simplicity)", ascending=False)

    x = np.arange(len(df_plot))
    width = 0.38

    plt.bar(x - width/2, df_plot["Pearson r (Simplicity)"], width, label="Pearson $r$ (CEFR Simplicity)", color="#2ca02c", alpha=0.9)
    plt.bar(x + width/2, df_plot["Spearman ρ (Rang)"], width, label="Spearman $\\rho$ (Rang)", color="#1f77b4", alpha=0.9)

    plt.xticks(x, df_plot["Modell"], rotation=35, ha="right", fontweight="medium", fontsize=10)
    plt.ylabel("Korrelationskoeffizient")
    plt.title("Evaluation aller Modelle auf MERLIN CEFR Benchmark (N=1.033 Dokumente)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.ylim(-0.15, 0.78)
    for i in range(len(df_plot)):
        plt.text(i - width/2, df_plot["Pearson r (Simplicity)"].iloc[i] + 0.015, f"{df_plot['Pearson r (Simplicity)'].iloc[i]:.3f}", ha="center", fontsize=8)
        plt.text(i + width/2, df_plot["Spearman ρ (Rang)"].iloc[i] + 0.015, f"{df_plot['Spearman ρ (Rang)'].iloc[i]:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    p1 = os.path.join(args.plot_dir, "merlin_correlation_barchart.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"Plot gespeichert: {p1}")

    # 2. Plot: Scatterplot-Grid für ALLE Modelle vs. CEFR Simplicity
    pred_cols = [c for c in df_out.columns if c.startswith("Pred_")]
    n_models = len(pred_cols)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5 * n_rows))
    axes = axes.flatten()

    colors = sns.color_palette("tab10", n_models)

    for i, col in enumerate(pred_cols):
        m_label = col.replace("Pred_", "").replace("_", " ")
        r_val, _ = pearsonr(df_out[col], df_out["cefr_simplicity"])
        rho_val, _ = spearmanr(df_out[col], df_out["cefr_simplicity"])

        sns.regplot(
            x=df_out["cefr_simplicity"], y=df_out[col], ax=axes[i],
            color=colors[i % len(colors)],
            scatter_kws={"alpha": 0.3, "s": 20},
            line_kws={"color": "#d62728", "lw": 1.8}
        )
        axes[i].set_title(f"{m_label}\n$r = {r_val:.3f}$ | $\\rho = {rho_val:.3f}$", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("CEFR Simplicity (1.0 = A1 ... 0.0 = C2)")
        axes[i].set_ylabel("Modell-Score")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Streudiagramme ALLER Modelle vs. CEFR-Niveau (MERLIN)", fontsize=15, fontweight="bold", y=1.002)
    plt.tight_layout()
    p2 = os.path.join(args.plot_dir, "merlin_scatter_all_models_grid.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"Plot gespeichert: {p2}")

    # 3. Plot: Monotonie-Boxplots über die 5 CEFR-Stufen (A1, A2, B1, B2, C1)
    cefr_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
    valid_levels = [lvl for lvl in cefr_order if lvl in df_out["cefr_level"].unique()]
    df_out["CEFR_Category"] = pd.Categorical(df_out["cefr_level"], categories=valid_levels, ordered=True)

    fig_box, axes_box = plt.subplots(n_rows, n_cols, figsize=(18, 4.5 * n_rows))
    axes_box = axes_box.flatten()

    for i, col in enumerate(pred_cols):
        m_label = col.replace("Pred_", "").replace("_", " ")
        sns.boxplot(
            data=df_out, x="CEFR_Category", y=col, ax=axes_box[i],
            palette="Blues_r", hue="CEFR_Category", legend=False, showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "6"}
        )
        axes_box[i].set_title(f"CEFR Monotonie: {m_label}", fontsize=11, fontweight="bold")
        axes_box[i].set_xlabel("CEFR-Stufe")
        axes_box[i].set_ylabel("Modell-Score")

    for j in range(i + 1, len(axes_box)):
        fig_box.delaxes(axes_box[j])

    plt.suptitle("Monotonie-Prüfung ALLER Modelle über CEFR-Stufen (MERLIN)", fontsize=15, fontweight="bold", y=1.002)
    plt.tight_layout()
    p3 = os.path.join(args.plot_dir, "merlin_boxplots_all_models_grid.png")
    plt.savefig(p3, dpi=150)
    plt.close()
    print(f"Plot gespeichert: {p3}")

    print("\n[ERFOLG] MERLIN CEFR Evaluation abgeschlossen.")


if __name__ == "__main__":
    main()
