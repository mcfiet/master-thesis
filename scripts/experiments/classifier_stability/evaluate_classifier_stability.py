#!/usr/bin/env python3
"""
scripts/experiments/classifier_stability/evaluate_classifier_stability.py

Aggregiert die Ergebnisse des Multi-Seed- und Kapazitäts-Stabilitätsexperiments:
1. Berechnet statistische Kennzahlen (Mean, Std, Min, Max, Spread) über alle Seeds.
2. Speichert JSON- und CSV-Zusammenfassungen in results/evaluation/classifier_stability/.
3. Erzeugt hochauflösende Visualisierungen in results/plots/experiments/classifier_stability/:
   - generalization_gap_learning_curves.png (In-Domain Val vs. OOD Lebenshilfe über Epochen)
   - multi_seed_boxplots.png (Seed-Varianz & Stabilitätsvergleich)
   - model_capacity_ablation.png (Modellgröße vs. Generalisierung)
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_generalization_gap(trajectories: Dict[str, Any], plot_dir: str):
    """
    Zeigt für jede Modellvariante den Verlauf von In-Domain Val Loss / Acc vs.
    Out-of-Domain Lebenshilfe Acc / Separation über die Epochen.
    """
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    models_to_plot = [m for m in ["art_256", "art_512", "art_1024", "sentence_model", "mixup_1024", "art_1024_tiny"] if m in trajectories]
    if not models_to_plot:
        models_to_plot = list(trajectories.keys())[:6]

    n_models = len(models_to_plot)
    cols = min(3, n_models)
    rows = int(np.ceil(n_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)
    axes = axes.flatten()

    pretty_names = {
        "art_256": "Artikel-Klassifikator (256 Tok)",
        "art_512": "Artikel-Klassifikator (512 Tok)",
        "art_1024": "Artikel-Klassifikator (1024 Tok)",
        "art_1024_tiny": "Artikel-1024 Tiny (180k Param)",
        "art_1024_medium": "Artikel-1024 Medium (700k Param)",
        "sentence_model": "Satz-Klassifikator (Maj. Vote)",
        "mixup_1024": "BiLSTM MixUp Regressor (1024 Tok)",
    }

    for idx, m_key in enumerate(models_to_plot):
        ax = axes[idx]
        m_data = trajectories[m_key]
        seeds = list(m_data.keys())

        if not seeds:
            continue

        # Epochen extrahieren
        n_epochs = len(m_data[seeds[0]])
        epochs = np.arange(1, n_epochs + 1)

        in_val_bacc_all = []
        ood_bacc_all = []
        ood_sep_all = []
        train_loss_all = []

        for s in seeds:
            history = m_data[s]
            in_val_bacc_all.append([h.get("val_bacc", 0.5) * 100 for h in history])
            ood_bacc_all.append([h.get("lh_bacc", 0.5) * 100 for h in history])
            ood_sep_all.append([h.get("lh_separation", 0.0) for h in history])
            train_loss_all.append([h.get("train_loss", 0.0) for h in history])

        in_val_mean = np.mean(in_val_bacc_all, axis=0)
        in_val_std = np.std(in_val_bacc_all, axis=0)

        ood_mean = np.mean(ood_bacc_all, axis=0)
        ood_std = np.std(ood_bacc_all, axis=0)

        # Plot In-Domain vs OOD
        ax.plot(epochs, in_val_mean, color="#2b5c8f", lw=2.2, label="In-Domain Val BAcc (%)")
        ax.fill_between(epochs, in_val_mean - in_val_std, in_val_mean + in_val_std, color="#2b5c8f", alpha=0.15)

        ax.plot(epochs, ood_mean, color="#2ca02c", lw=2.5, label="OOD Lebenshilfe BAcc (%)")
        ax.fill_between(epochs, ood_mean - ood_std, ood_mean + ood_std, color="#2ca02c", alpha=0.2)

        ax.axhline(50, color="gray", linestyle=":", alpha=0.7, label="Chance Level (50%)")

        ax.set_title(pretty_names.get(m_key, m_key), fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoche")
        ax.set_ylabel("Balanced Accuracy (%)")
        ax.set_ylim(40, 103)
        ax.legend(loc="lower right" if "art" in m_key else "lower right", fontsize=8.5)

    # Leere Achsen ausblenden
    for j in range(len(models_to_plot), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Generalization Gap & Overfitting-Trajektorien über 30 Epochen (Mittelwert ± Std über 5 Seeds)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, "generalization_gap_learning_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Generalization-Gap-Plot gespeichert: {save_path}")


def plot_seed_boxplots(summary_df: pd.DataFrame, plot_dir: str):
    """
    Zeigt die Varianz von Balanced Accuracy und Separation über Seeds als Boxplots.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    pretty_names = {
        "art_256": "Artikel 256",
        "art_512": "Artikel 512",
        "art_1024": "Artikel 1024",
        "art_1024_tiny": "Artikel 1024 (Tiny)",
        "art_1024_medium": "Artikel 1024 (Medium)",
        "sentence_model": "Satzmodell (MajVote)",
        "mixup_1024": "MixUp Regressor 1024",
    }

    df_plot = summary_df.copy()
    df_plot["Modellname"] = df_plot["model"].map(lambda x: pretty_names.get(x, x))
    df_plot["OOD BAcc (%)"] = df_plot["lh_bacc"] * 100
    df_plot["OOD AUC"] = df_plot["lh_auc"]
    df_plot["Separation Δ"] = df_plot["lh_separation"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Balanced Accuracy Boxplot
    sns.boxplot(data=df_plot, x="Modellname", y="OOD BAcc (%)", ax=axes[0], palette="Set2", showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 7})
    sns.stripplot(data=df_plot, x="Modellname", y="OOD BAcc (%)", ax=axes[0], color="black", size=6, jitter=0.2, alpha=0.7)
    axes[0].set_title("OOD Balanced Accuracy über 5 Seeds (Lebenshilfe)", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Balanced Accuracy (%)")
    axes[0].tick_params(axis="x", rotation=30)

    # 2. Separation Boxplot
    sns.boxplot(data=df_plot, x="Modellname", y="Separation Δ", ax=axes[1], palette="Set2", showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 7})
    sns.stripplot(data=df_plot, x="Modellname", y="Separation Δ", ax=axes[1], color="black", size=6, jitter=0.2, alpha=0.7)
    axes[1].set_title("OOD Klassenseparation Δ (LS - AS)", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Separation Δ")
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle("Stabilitätsvergleich über 5 Zufalls-Seeds (Lebenshilfe Goldstandard)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, "multi_seed_boxplots.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Multi-Seed Boxplots gespeichert: {save_path}")


def generate_summary_tables(summary_df: pd.DataFrame, output_dir: str):
    """
    Berechnet die aggregierten Mittelwerte und Standardabweichungen.
    """
    pretty_names = {
        "art_256": "BiLSTM Artikel-Klassifikator (256)",
        "art_512": "BiLSTM Artikel-Klassifikator (512)",
        "art_1024": "BiLSTM Artikel-Klassifikator (1024)",
        "art_1024_tiny": "BiLSTM Artikel-Klassifikator (1024 Tiny)",
        "art_1024_medium": "BiLSTM Artikel-Klassifikator (1024 Medium)",
        "sentence_model": "BiLSTM Satz-Klassifikator (MajVote)",
        "mixup_1024": "BiLSTM MixUp Regressor (1024)",
    }

    results = []
    for model_name, group in summary_df.groupby("model"):
        n_seeds = len(group)
        bacc_mean, bacc_std = group["lh_bacc"].mean() * 100, group["lh_bacc"].std() * 100
        auc_mean, auc_std = group["lh_auc"].mean(), group["lh_auc"].std()
        sep_mean, sep_std = group["lh_separation"].mean(), group["lh_separation"].std()
        ls_mean, ls_std = group["lh_ls_mean"].mean(), group["lh_ls_mean"].std()
        as_mean, as_std = group["lh_as_mean"].mean(), group["lh_as_mean"].std()
        pm_mean, pm_std = group["lh_pair_match"].mean() * 100, group["lh_pair_match"].std() * 100

        results.append({
            "Modell": pretty_names.get(model_name, model_name),
            "Seeds": n_seeds,
            "Ø Score (LS)": f"{ls_mean:.3f} ± {ls_std:.2f}",
            "Ø Score (AS)": f"{as_mean:.3f} ± {as_std:.2f}",
            "Separation (Δ)": f"{sep_mean:.3f} ± {sep_std:.2f}",
            "Balanced Acc": f"{bacc_mean:.2f}% ± {bacc_std:.2f}%",
            "BAcc (Min - Max)": f"{group['lh_bacc'].min()*100:.1f}% - {group['lh_bacc'].max()*100:.1f}%",
            "ROC-AUC": f"{auc_mean:.4f} ± {auc_std:.4f}",
            "Pair Match": f"{pm_mean:.1f}% ± {pm_std:.1f}%",
            "_raw_bacc_mean": bacc_mean,
            "_raw_bacc_std": bacc_std,
            "_raw_auc_mean": auc_mean,
            "_raw_model_key": model_name
        })

    # Sortieren nach BAcc
    results_df = pd.DataFrame(results).sort_values(by="_raw_bacc_mean", ascending=False)

    json_path = os.path.join(output_dir, "classifier_stability_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, "classifier_stability_summary.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"[OK] Zusammenfassung gespeichert: {json_path} und {csv_path}")
    print("\n" + "=" * 80)
    print("ERGEBNIS-TABELLE (STABILITÄTS-BENCHMARK):")
    print("=" * 80)
    print(results_df[["Modell", "Balanced Acc", "BAcc (Min - Max)", "ROC-AUC", "Pair Match"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Classifier Stability Experiment")
    parser.add_argument("--eval_dir", default="results/evaluation/classifier_stability")
    parser.add_argument("--plot_dir", default="results/plots/experiments/classifier_stability")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)

    traj_path = os.path.join(args.eval_dir, "epoch_trajectories.json")
    summary_raw_path = os.path.join(args.eval_dir, "seed_summary_raw.csv")

    if not os.path.exists(traj_path) or not os.path.exists(summary_raw_path):
        print(f"[WARNUNG] Trajektoriendatei ({traj_path}) oder Roh-Zusammenfassung nicht gefunden.")
        print("Führe zuerst 'train_and_track_stability.py' aus.")
        return

    with open(traj_path, "r", encoding="utf-8") as f:
        trajectories = json.load(f)

    summary_df = pd.read_csv(summary_raw_path)

    # 1. Plots generieren
    plot_generalization_gap(trajectories, args.plot_dir)
    plot_seed_boxplots(summary_df, args.plot_dir)

    # 2. Aggregierte Tabellen generieren
    generate_summary_tables(summary_df, args.eval_dir)


if __name__ == "__main__":
    main()
