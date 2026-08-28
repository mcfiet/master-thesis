#!/usr/bin/env python3
"""
scripts/evaluation/evaluate_textcomplexityde.py

Umfassende Evaluation ALLER Modellparadigmen auf dem externen Benchmark TextComplexityDE
(Naderi et al., 2019: 1.000 deutsche Wikipedia-Sätze mit menschlichen Urteilen):

1. Regressoren:
   - BiLSTM Satz-Regressor (Satz-Ebene)
   - BiLSTM MixUp Regressoren (256, 512, 1024 Tokens)
2. Klassifikatoren:
   - BiLSTM Satz-Klassifikator (P(LS) auf Satzebene)
   - BiLSTM Artikel-Klassifikatoren (256, 512, 1024 Tokens)
3. Traditionelle Baselines:
   - Flesch Reading Ease
   - Wiener Sachtextformel
   - LIX Lesbarkeitsindex

Berechnet Pearson r, Spearman rho, Kendall tau gegen:
- MOS_Complexity (invertiert als Simplicity)
- MOS_Understandability
- MOS_Lexical_difficulty
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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


def evaluate_series(preds: np.ndarray, y_simp: np.ndarray, y_comp: np.ndarray, y_und: np.ndarray, y_lex: np.ndarray, name: str, cat: str) -> Dict[str, Any]:
    # Pearson & Spearman mit menschlicher Simplicity (1.0 = einfach)
    r_simp, p_simp = pearsonr(preds, y_simp)
    rho_simp, _ = spearmanr(preds, y_simp)
    tau_simp, _ = kendalltau(preds, y_simp)

    # Korrelation mit Verständlichkeit (positiv) & lexikalischer Schwierigkeit (negativ)
    r_und, _ = pearsonr(preds, y_und)
    r_lex, _ = pearsonr(preds, -y_lex)  # negiert, da hohe lex. difficulty = geringe Simplicity

    # MSE & MAE auf normalisierter Skala
    # Skaliere Vorhersagen auf [0, 1] Min-Max für fairen MAE/RMSE
    preds_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
    rmse = float(np.sqrt(mean_squared_error(y_simp, preds_norm)))
    mae = float(mean_absolute_error(y_simp, preds_norm))

    return {
        "Modell": name,
        "Kategorie": cat,
        "Pearson r (Simplicity)": float(r_simp),
        "Spearman ρ (Rang)": float(rho_simp),
        "Kendall τ": float(tau_simp),
        "Pearson r (Verständlichkeit)": float(r_und),
        "Pearson r (Lexikalisch)": float(r_lex),
        "RMSE (normiert)": float(rmse),
        "MAE (normiert)": float(mae),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_csv", default="data/analysis/textcomplexityde/ratings.csv")
    parser.add_argument("--output_csv", default="results/evaluation/textcomplexityde_all_models_eval.csv")
    parser.add_argument("--summary_json", default="results/evaluation/textcomplexityde_summary.json")
    parser.add_argument("--plot_dir", default="results/plots/experiments/textcomplexityde")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device}")

    nlp = spacy.blank("de")

    # Benchmark-Daten laden
    if not os.path.exists(args.benchmark_csv):
        os.makedirs(os.path.dirname(args.benchmark_csv), exist_ok=True)
        url = "https://raw.githubusercontent.com/babaknaderi/TextComplexityDE/master/data/ratings.csv"
        print(f"Lade TextComplexityDE von {url}...")
        urllib.request.urlretrieve(url, args.benchmark_csv)

    try:
        df_raw = pd.read_csv(args.benchmark_csv, encoding="utf-8")
    except UnicodeDecodeError:
        df_raw = pd.read_csv(args.benchmark_csv, encoding="latin-1")

    # Aggregation pro Satz
    if "Sentence" in df_raw.columns and "MOS_Complexity" in df_raw.columns:
        df_sentences = df_raw.groupby("Sentence")[["MOS_Complexity", "MOS_Understandability", "MOS_Lexical_difficulty"]].mean().reset_index()
    else:
        df_sentences = df_raw

    sentences = df_sentences["Sentence"].tolist()
    y_comp = df_sentences["MOS_Complexity"].to_numpy()
    y_und = df_sentences["MOS_Understandability"].to_numpy()
    y_lex = df_sentences["MOS_Lexical_difficulty"].to_numpy()

    # Simplicity: Invertierte Komplexität auf [0, 1]
    y_simp = 1.0 - ((y_comp - y_comp.min()) / (y_comp.max() - y_comp.min() + 1e-8))

    df_out = df_sentences.copy()
    df_out["Human_Simplicity"] = y_simp

    # Baselines berechnen
    print("Berechne traditionelle Lesbarkeitsbaselines...")
    flesch_scores = np.array([textstat.flesch_reading_ease(str(s)) for s in sentences])
    wiener_scores = np.array([-textstat.wiener_sachtextformel(str(s), 1) for s in sentences])  # negiert
    lix_scores = np.array([-textstat.lix(str(s)) for s in sentences])  # negiert

    df_out["Pred_Flesch"] = flesch_scores
    df_out["Pred_Wiener"] = wiener_scores
    df_out["Pred_LIX"] = lix_scores

    # Modelle definieren
    models_config = [
        # 1. Regressoren (Längenablation 256, 512, 1024)
        ("BiLSTM MixUp Regressor (256)", "results/models/regressor_length_exp/bilstm_mixup_regression_256.pt", "data/regressor_length_exp/mixup_vocab_256.json", "reg", 256, "Regressor (256)"),
        ("BiLSTM MixUp Regressor (512)", "results/models/regressor_length_exp/bilstm_mixup_regression_512.pt", "data/regressor_length_exp/mixup_vocab_512.json", "reg", 512, "Regressor (512)"),
        ("BiLSTM MixUp Regressor (1024)", "results/models/bilstm_mixup_regression.pt", "data/vocabs/mixup_vocab.json", "reg", 1024, "Regressor (1024)"),
        
        # 2. Klassifikatoren (P(LS) auf Satz- und Dokumentebene)
        ("BiLSTM Satz-Klassifikator", "results/models/bilstm_sentence_classifier.pt", "data/vocabs/sentence_vocab.json", "clf", 100, "Klassifikator (Satz)"),
        ("BiLSTM Artikel-Klassifikator (256)", "results/models/classifier_length_exp/bilstm_article_classifier_256.pt", "data/classifier_length_exp/article_vocab_256.json", "clf", 256, "Klassifikator (256)"),
        ("BiLSTM Artikel-Klassifikator (512)", "results/models/classifier_length_exp/bilstm_article_classifier_512.pt", "data/classifier_length_exp/article_vocab_512.json", "clf", 512, "Klassifikator (512)"),
        ("BiLSTM Artikel-Klassifikator (1024)", "results/models/bilstm_article_classifier.pt", "data/vocabs/article_vocab.json", "clf", 1024, "Klassifikator (1024)"),
    ]

    summary_records = []

    # Baselines evaluieren
    summary_records.append(evaluate_series(flesch_scores, y_simp, y_comp, y_und, y_lex, "Flesch Reading Ease", "Baseline"))
    summary_records.append(evaluate_series(wiener_scores, y_simp, y_comp, y_und, y_lex, "Wiener Sachtextformel", "Baseline"))
    summary_records.append(evaluate_series(lix_scores, y_simp, y_comp, y_und, y_lex, "LIX Lesbarkeitsindex", "Baseline"))

    # Neuronale Modelle evaluieren
    for name, m_path, v_path, m_type, max_len, cat in models_config:
        # Fallback falls Pfad in token_length_exp liegt
        if not os.path.exists(m_path):
            alt_m = m_path.replace("regressor_length_exp", "token_length_exp").replace("classifier_length_exp", "token_length_exp")
            if os.path.exists(alt_m): m_path = alt_m
        if not os.path.exists(v_path):
            alt_v = v_path.replace("regressor_length_exp", "token_length_exp").replace("classifier_length_exp", "token_length_exp")
            if os.path.exists(alt_v): v_path = alt_v

        if not os.path.exists(m_path) or not os.path.exists(v_path):
            print(f"[SKIP] {name}: Checkpoint nicht gefunden ({m_path})")
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

        tokenized = []
        for s in sentences:
            doc = nlp(str(s or ""))
            toks = [t.text.lower() for t in doc if not t.is_space]
            enc = [vocab.get(t, 1) for t in toks][:max_len]
            if not enc: enc = [0]
            tokenized.append(enc)

        padded = np.zeros((len(sentences), max_len), dtype=np.int64)
        for i, seq in enumerate(tokenized):
            padded[i, :len(seq)] = seq

        tensor_x = torch.tensor(padded, dtype=torch.long, device=device)
        with torch.no_grad():
            if m_type == "reg":
                preds = model(tensor_x).cpu().numpy()
            else:
                preds = torch.sigmoid(model(tensor_x)).cpu().numpy()

        df_out[f"Pred_{name.replace(' ', '_')}"] = preds
        summary_records.append(evaluate_series(preds, y_simp, y_comp, y_und, y_lex, name, cat))

    df_out.to_csv(args.output_csv, index=False)
    print(f"Detail-Vorhersagen gespeichert: {args.output_csv}")

    df_summary = pd.DataFrame(summary_records)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("TEXTCOMPLEXITYDE BENCHMARK MASTER TABELLE (N=1.000 Wikipedia-Sätze)")
    print("=" * 90)
    cols_show = ["Modell", "Kategorie", "Pearson r (Simplicity)", "Spearman ρ (Rang)", "Pearson r (Verständlichkeit)", "MAE (normiert)"]
    print(df_summary[cols_show].to_string(index=False))

    # PLOTS GENERIEREN
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    # 1. Plot: Barchart der Pearson & Spearman Korrelationen
    plt.figure(figsize=(14, 6))
    df_plot = df_summary.sort_values(by="Pearson r (Simplicity)", ascending=False)
    
    x = np.arange(len(df_plot))
    width = 0.38

    plt.bar(x - width/2, df_plot["Pearson r (Simplicity)"], width, label="Pearson $r$ (Simplicity)", color="#2ca02c", alpha=0.9)
    plt.bar(x + width/2, df_plot["Spearman ρ (Rang)"], width, label="Spearman $\\rho$ (Rang)", color="#1f77b4", alpha=0.9)

    plt.xticks(x, df_plot["Modell"], rotation=35, ha="right", fontweight="medium", fontsize=10)
    plt.ylabel("Korrelationskoeffizient")
    plt.title("Evaluation aller Modelle auf TextComplexityDE (N=1.000 Sätze)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.ylim(0, 0.78)
    for i in range(len(df_plot)):
        plt.text(i - width/2, df_plot["Pearson r (Simplicity)"].iloc[i] + 0.012, f"{df_plot['Pearson r (Simplicity)'].iloc[i]:.3f}", ha="center", fontsize=8)
        plt.text(i + width/2, df_plot["Spearman ρ (Rang)"].iloc[i] + 0.012, f"{df_plot['Spearman ρ (Rang)'].iloc[i]:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    p1 = os.path.join(args.plot_dir, "textcomplexityde_correlation_barchart.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Plot gespeichert: {p1}")

    # 2. Plot: Umfassendes Scatterplot-Grid für ALLE Modelle vs. Human Simplicity
    pred_cols = [c for c in df_out.columns if c.startswith("Pred_")]
    n_models = len(pred_cols)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5 * n_rows))
    axes = axes.flatten()

    colors = sns.color_palette("tab10", n_models)

    for i, col in enumerate(pred_cols):
        m_label = col.replace("Pred_", "").replace("_", " ")
        r_val, _ = pearsonr(df_out[col], df_out["Human_Simplicity"])
        rho_val, _ = spearmanr(df_out[col], df_out["Human_Simplicity"])
        
        sns.regplot(
            x=df_out["Human_Simplicity"], y=df_out[col], ax=axes[i],
            color=colors[i % len(colors)],
            scatter_kws={'alpha': 0.3, 's': 20},
            line_kws={'color': '#d62728', 'lw': 1.8}
        )
        axes[i].set_title(f"{m_label}\n$r = {r_val:.3f}$ | $\\rho = {rho_val:.3f}$", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Menschliche Simplicity (1 = Leicht)")
        axes[i].set_ylabel(f"Modell-Score")

    # Verbleibende leere Subplots ausblenden
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Streudiagramme ALLER Modelle vs. Menschliche Urteile (TextComplexityDE)", fontsize=15, fontweight="bold", y=1.002)
    plt.tight_layout()
    p2 = os.path.join(args.plot_dir, "textcomplexityde_scatter_all_models_grid.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Plot gespeichert: {p2}")

    # 3. Plot: Monotonie-Boxplots für ALLE Modelle über die 5 menschlichen Komplexitätsstufen
    bins = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
    labels = ["1: Sehr einfach", "2: Einfach", "3: Mittel", "4: Schwer", "5: Sehr schwer"]
    df_out["Complexity_Category"] = pd.cut(df_out["MOS_Complexity"], bins=bins, labels=labels, include_lowest=True)

    fig_box, axes_box = plt.subplots(n_rows, n_cols, figsize=(18, 4.5 * n_rows))
    axes_box = axes_box.flatten()

    for i, col in enumerate(pred_cols):
        m_label = col.replace("Pred_", "").replace("_", " ")
        sns.boxplot(
            data=df_out, x="Complexity_Category", y=col, ax=axes_box[i],
            palette="Blues_r", hue="Complexity_Category", legend=False, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"6"}
        )
        axes_box[i].set_title(f"Monotonie: {m_label}", fontsize=11, fontweight="bold")
        axes_box[i].set_xlabel("Menschliche Komplexitätsstufe")
        axes_box[i].set_ylabel("Modell-Score")
        axes_box[i].tick_params(axis='x', rotation=25)

    for j in range(i + 1, len(axes_box)):
        fig_box.delaxes(axes_box[j])

    plt.suptitle("Monotonie-Prüfung ALLER Modelle über menschliche Komplexitätsstufen", fontsize=15, fontweight="bold", y=1.002)
    plt.tight_layout()
    p3 = os.path.join(args.plot_dir, "textcomplexityde_boxplots_all_models_grid.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Plot gespeichert: {p3}")

    print("\n[ERFOLG] TextComplexityDE Evaluation abgeschlossen.")


if __name__ == "__main__":
    main()
