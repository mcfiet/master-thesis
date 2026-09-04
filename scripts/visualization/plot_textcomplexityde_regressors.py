#!/usr/bin/env python3
"""
scripts/visualization/plot_textcomplexityde_regressors.py

Generiert aktualisierte Plots für TextComplexityDE (N=1.000 Sätze)
nur für Regressoren und traditionelle Baselines (ohne Klassifikatoren).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

def main():
    csv_path = "results/evaluation/textcomplexityde_all_models_eval.csv"
    if not os.path.exists(csv_path):
        csv_path = "results/evaluation/textcomplexityde_all_models_eval.csv"

    df = pd.read_csv(csv_path)
    print(f"Geladene CSV: {csv_path} ({len(df)} Zeilen)")

    # Ausgewählte Spalten: Nur Baselines und Regressoren
    selected_cols = [
        ("Pred_Flesch", "Flesch Reading Ease", "Baseline"),
        ("Pred_Wiener", "Wiener Sachtextformel", "Baseline"),
        ("Pred_LIX", "LIX Lesbarkeitsindex", "Baseline"),
        ("Pred_BiLSTM_MixUp_Regressor_(256)", "BiLSTM MixUp Regressor (256)", "Regressor (256)"),
        ("Pred_BiLSTM_MixUp_Regressor_(512)", "BiLSTM MixUp Regressor (512)", "Regressor (512)"),
        ("Pred_BiLSTM_MixUp_Regressor_(1024)", "BiLSTM MixUp Regressor (1024)", "Regressor (1024)"),
    ]

    # Nur vorhandene Spalten nehmen
    valid_cols = [(col, label, cat) for col, label, cat in selected_cols if col in df.columns]

    y_simp = df["Human_Simplicity"].values if "Human_Simplicity" in df.columns else 1.0 - (df["MOS_Complexity"] - 1.0)/6.0

    summary_records = []
    for col, label, cat in valid_cols:
        r_val, _ = pearsonr(df[col], y_simp)
        rho_val, _ = spearmanr(df[col], y_simp)
        summary_records.append({
            "Modell": label,
            "Kategorie": cat,
            "Col": col,
            "Pearson r": r_val,
            "Spearman rho": rho_val
        })

    df_summary = pd.DataFrame(summary_records)
    print(df_summary[["Modell", "Pearson r", "Spearman rho"]])

    # Plot styling
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    target_dirs = [
        "results/plots/experiments/textcomplexityde",
        "results/plots/experiments/textcomplexityde",
        "thesis/images"
    ]
    for d in target_dirs:
        os.makedirs(d, exist_ok=True)

    # 1. BARCHART (6 Modelle: 3 Baselines + 3 Regressoren)
    plt.figure(figsize=(11, 5.5))
    df_plot = df_summary.sort_values(by="Pearson r", ascending=False).reset_index(drop=True)

    x = np.arange(len(df_plot))
    width = 0.36

    plt.bar(x - width/2, df_plot["Pearson r"], width, label="Pearson $r$ (Simplicity)", color="#2ca02c", alpha=0.9)
    plt.bar(x + width/2, df_plot["Spearman rho"], width, label="Spearman $\\rho$ (Rang)", color="#1f77b4", alpha=0.9)

    plt.xticks(x, df_plot["Modell"], rotation=25, ha="right", fontweight="medium", fontsize=11)
    plt.ylabel("Korrelationskoeffizient", fontsize=12)
    plt.title("Evaluation der Regressoren und Baselines auf TextComplexityDE (N=1.000 Sätze)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True, fontsize=11)
    plt.ylim(0, 0.78)

    for i in range(len(df_plot)):
        plt.text(i - width/2, df_plot["Pearson r"].iloc[i] + 0.012, f"{df_plot['Pearson r'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
        plt.text(i + width/2, df_plot["Spearman rho"].iloc[i] + 0.012, f"{df_plot['Spearman rho'].iloc[i]:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    for d in target_dirs:
        plt.savefig(os.path.join(d, "textcomplexityde_correlation_barchart.png"), dpi=300)
    plt.close()
    print("-> textcomplexityde_correlation_barchart.png gespeichert")

    # 2. SCATTERPLOT GRID (2 Zeilen x 3 Spalten = 6 Subplots)
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
        axes[i].set_xlabel("Menschliche Simplicity (1 = Leicht)", fontsize=10.5)
        axes[i].set_ylabel("Modell-Score", fontsize=10.5)

    plt.suptitle("Streudiagramme der Regressoren und Baselines vs. Menschliche Urteile (TextComplexityDE)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    for d in target_dirs:
        plt.savefig(os.path.join(d, "textcomplexityde_scatter_all_models_grid.png"), dpi=300)
    plt.close()
    print("-> textcomplexityde_scatter_all_models_grid.png gespeichert")

    # 3. BOXPLOT GRID (2 Zeilen x 3 Spalten = 6 Subplots)
    bins = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
    labels = ["1: Sehr einfach", "2: Einfach", "3: Mittel", "4: Schwer", "5: Sehr schwer"]
    df["Complexity_Category"] = pd.cut(df["MOS_Complexity"], bins=bins, labels=labels, include_lowest=True)

    fig_box, axes_box = plt.subplots(2, 3, figsize=(16, 10))
    axes_box = axes_box.flatten()

    for i, (col, label, cat) in enumerate(valid_cols):
        sns.boxplot(
            data=df, x="Complexity_Category", y=col, ax=axes_box[i],
            palette="Blues_r", hue="Complexity_Category", legend=False, showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"7"}
        )
        axes_box[i].set_title(f"Monotonie: {label}", fontsize=12, fontweight="bold")
        axes_box[i].set_xlabel("Menschliche Komplexitätsstufe", fontsize=10.5)
        axes_box[i].set_ylabel("Modell-Score", fontsize=10.5)
        axes_box[i].tick_params(axis='x', rotation=22)

    plt.suptitle("Monotonie-Prüfung der Regressoren und Baselines über menschliche Komplexitätsstufen", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    for d in target_dirs:
        plt.savefig(os.path.join(d, "textcomplexityde_boxplots_all_models_grid.png"), dpi=300)
    plt.close()
    print("-> textcomplexityde_boxplots_all_models_grid.png gespeichert")

if __name__ == "__main__":
    main()
