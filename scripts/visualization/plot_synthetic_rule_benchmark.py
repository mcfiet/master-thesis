#!/usr/bin/env python3
import os
import sys
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv", default="results/evaluation/synthetic_rule_benchmark_256_eval.csv")
    parser.add_argument("--summary_json", default="results/evaluation/synthetic_rule_benchmark_256_summary.json")
    parser.add_argument("--plot_dir", default="results/plots/experiments/rule_sensitivity")
    parser.add_argument("--thesis_dir", default="thesis/images")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.thesis_dir, exist_ok=True)

    if not os.path.exists(args.eval_csv):
        print("CSV nicht gefunden: " + str(args.eval_csv))
        return

    df = pd.read_csv(args.eval_csv)

    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = sns.color_palette("Set2")

    var_order = ["base_ls", "subjunctive_100", "passive_100", "genitive_100", "combined_all", "control_synonyms"]
    var_labels = ["Basis (LS)", "Konjunktiv\n(100% Verben)", "Passiv\n(100% Sätze)", "Genitiv\n(100% Dative)", "Kombiniert\n(Alle Regeln)", "Kontrolle\n(Synonyme)"]
    label_map = dict(zip(var_order, var_labels))

    # 1. Balkendiagramm der Metrik-Deltas vs. Basis
    df_non_base = df[df["variant"] != "base_ls"].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ("delta_flesch_de", "Δ Flesch Reading Ease (DE)", axes[0], palette[0]),
        ("delta_lix", "Δ LIX (Lesbarkeitsindex)", axes[1], palette[1]),
        ("delta_wstf_4", "Δ Wiener Sachtextformel (WSTF 4)", axes[2], palette[2]),
    ]

    for col, title, ax, col_color in metrics:
        sns.barplot(data=df_non_base, x="variant", y=col, order=var_order[1:], ax=ax, color=col_color, ci="sd", capsize=0.1)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("Differenz zur perfekten Basis (LS)", fontsize=10)
        ax.set_xlabel("")
        ax.set_xticklabels([label_map[v] for v in var_order[1:]], fontsize=9)

    plt.suptitle("Auswirkung flächendeckender Grammatikverstöße auf Absatzerhebung (100–256 Tokens)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_bars = os.path.join(args.plot_dir, "synthetic_rule_benchmark_256_deltas.png")
    plt.savefig(out_bars, dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(args.thesis_dir, "synthetic_rule_benchmark_256_deltas.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Gespeichert: " + str(out_bars))

    # 2. Parallel Coordinates Trajektorie pro Text
    plt.figure(figsize=(12, 6))
    for t_id in df["text_id"].unique():
        df_t = df[df["text_id"] == t_id].set_index("variant").reindex(var_order).reset_index()
        plt.plot(df_t["variant"], df_t["flesch_de"], marker="o", linewidth=2.0, alpha=0.8, label=t_id)

    plt.title("Trajektorie des Flesch Reading Ease über 100%ige Regelstörungs-Stufen", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Störungs-Variante (100–256 Tokens)", fontsize=11, fontweight="bold")
    plt.ylabel("Flesch Reading Ease (DE)", fontsize=11, fontweight="bold")
    plt.xticks(range(len(var_order)), var_labels, fontsize=10)
    plt.legend(title="Text-ID", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out_traj = os.path.join(args.plot_dir, "synthetic_rule_benchmark_256_trajectories.png")
    plt.savefig(out_traj, dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(args.thesis_dir, "synthetic_rule_benchmark_256_trajectories.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Gespeichert: " + str(out_traj))

if __name__ == "__main__":
    main()
