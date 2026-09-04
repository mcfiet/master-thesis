#!/usr/bin/env python3
"""
scripts/visualization/plot_merlin_regressors.py

Generiert aktualisierte Plots für MERLIN (N=1.033 Texte)
nur für Regressoren und traditionelle Baselines (ohne Klassifikatoren).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

def main():
    csv_path = "results/evaluation/merlin_all_models_eval.csv"
    df = pd.read_csv(csv_path)
    print(f"Geladene CSV: {csv_path} ({len(df)} Zeilen)")

    # Ausgewählte Spalten: Nur Baselines und Regressoren (6 Modelle)
    selected_cols = [
        ("Pred_Flesch", "Flesch Reading Ease", "Baseline"),
        ("Pred_Wiener", "Wiener Sachtextformel", "Baseline"),
        ("Pred_LIX", "LIX Lesbarkeitsindex", "Baseline"),
        ("Pred_BiLSTM_MixUp_Regressor_(256)", "BiLSTM MixUp Regressor (256)", "Regressor (256)"),
        ("Pred_BiLSTM_MixUp_Regressor_(512)", "BiLSTM MixUp Regressor (512)", "Regressor (512)"),
        ("Pred_BiLSTM_MixUp_Regressor_(1024)", "BiLSTM MixUp Regressor (1024)", "Regressor (1024)"),
    ]

    valid_cols = [(col, label, cat) for col, label, cat in selected_cols if col in df.columns]

    y_simp = df["cefr_simplicity"].values

    # Plot styling
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    target_dirs = [
        "results/plots/experiments/merlin",
        "thesis/images"
    ]
    for d in target_dirs:
        os.makedirs(d, exist_ok=True)

    # 1. SCATTERPLOT GRID (2 Zeilen x 3 Spalten = 6 Subplots)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.flatten()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (col, label, cat) in enumerate(valid_cols):
        r_val, _ = pearsonr(df[col], y_simp)
        rho_val, _ = spearmanr(df[col], y_simp)

        sns.regplot(
            x=y_simp, y=df[col], ax=axes[i],
            color=colors[i % len(colors)],
            scatter_kws={'alpha': 0.25, 's': 18},
            line_kws={'color': '#d62728', 'lw': 2.0}
        )
        axes[i].set_title(f"{label}\n$r = {r_val:.3f}$ | $\\rho = {rho_val:.3f}$", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("CEFR Simplicity (1.0 = A1 ... 0.0 = C2)", fontsize=10.5)
        axes[i].set_ylabel("Modell-Score", fontsize=10.5)

    plt.suptitle("Streudiagramme der Regressoren und Baselines vs. CEFR-Niveau (MERLIN)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    for d in target_dirs:
        plt.savefig(os.path.join(d, "merlin_scatter_all_models_grid.png"), dpi=300)
    plt.close()
    print("-> merlin_scatter_all_models_grid.png gespeichert")

    # 2. BOXPLOT GRID (2 Zeilen x 3 Spalten = 6 Subplots)
    cefr_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
    valid_levels = [lvl for lvl in cefr_order if lvl in df["cefr_level"].unique()]
    df["CEFR_Category"] = pd.Categorical(df["cefr_level"], categories=valid_levels, ordered=True)

    fig_box, axes_box = plt.subplots(2, 3, figsize=(16, 10))
    axes_box = axes_box.flatten()

    for i, (col, label, cat) in enumerate(valid_cols):
        sns.boxplot(
            data=df, x="CEFR_Category", y=col, ax=axes_box[i],
            palette="Blues_r", hue="CEFR_Category", legend=False, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"7"}
        )
        axes_box[i].set_title(f"CEFR Monotonie: {label}", fontsize=12, fontweight="bold")
        axes_box[i].set_xlabel("CEFR-Stufe", fontsize=10.5)
        axes_box[i].set_ylabel("Modell-Score", fontsize=10.5)
        axes_box[i].tick_params(axis='x', rotation=0)

    plt.suptitle("Monotonie-Prüfung der Regressoren und Baselines über CEFR-Stufen (MERLIN)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    for d in target_dirs:
        plt.savefig(os.path.join(d, "merlin_boxplots_all_models_grid.png"), dpi=300)
    plt.close()
    print("-> merlin_boxplots_all_models_grid.png gespeichert")

if __name__ == "__main__":
    main()
