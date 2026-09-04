#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Similarity Threshold Experiment: Master Evaluation & Consolidation Script
=============================================================================
Sammelt alle Metriken und Detailvorhersagen aus:
- MixUp Regressor (0.60, 0.70, 0.80 bis 0.98)
- SFT mBART-50 (0.60, 0.70, 0.80 bis 0.98)

Erstellt konsolidierte Tabellen, wissenschaftliche Vergleichsplots und
LaTeX-Tabellen für die Masterarbeit.
=============================================================================
"""

import argparse
import glob
import json
import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("EvaluateSimilarityAblation")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate & Consolidate Similarity Threshold Experiments")
    parser.add_argument('--results_dir', default="results/experiments/similarity_threshold")
    parser.add_argument('--plots_dir', default="results/experiments/similarity_threshold/plots")
    parser.add_argument('--corpus_path', default="data/analysis/corpus_master.csv")
    parser.add_argument('--output_summary_csv', default="results/experiments/similarity_threshold/similarity_threshold_summary.csv")
    parser.add_argument('--output_details_csv', default="results/experiments/similarity_threshold/similarity_threshold_details.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 300

    logger.info("=== Konsolidiere Similarity Threshold Experimente ===")

    # 1. Korpus-Statistiken über die Schwellenwerte ermitteln
    corpus_stats = []
    if os.path.exists(args.corpus_path):
        df_corpus = pd.read_csv(args.corpus_path)
        for s_min in [0.60, 0.70, 0.80, 0.85, 0.90]:
            sub = df_corpus[(df_corpus["semantic_similarity_8192"] >= s_min) & (df_corpus["semantic_similarity_8192"] <= 0.98)]
            corpus_stats.append({
                "min_sim": s_min,
                "max_sim": 0.98,
                "pairs": len(sub),
                "retention_pct": round((len(sub) / len(df_corpus)) * 100.0, 2),
                "as_tokens": int(sub["as_tokens"].sum()) if "as_tokens" in sub.columns else 0,
                "ls_tokens": int(sub["ls_tokens"].sum()) if "ls_tokens" in sub.columns else 0,
                "token_ratio": round(sub["ls_tokens"].sum() / max(1, sub["as_tokens"].sum()), 4) if "as_tokens" in sub.columns else 0
            })
    df_corpus_stats = pd.DataFrame(corpus_stats)

    # 2. JSON Metrics Dateien laden
    json_files = sorted(glob.glob(os.path.join(args.results_dir, "*_metrics.json")))
    logger.info(f"Gefundene Metriken-Dateien: {len(json_files)}")

    all_metrics = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_metrics.append(data)

    df_metrics = pd.DataFrame(all_metrics)
    if not df_metrics.empty:
        df_metrics.to_csv(args.output_summary_csv, index=False)
        logger.info(f"Zusammenfassung gespeichert unter: {args.output_summary_csv}")
    else:
        logger.warning("Keine Metriken-Dateien gefunden.")

    # 3. Detail CSVs laden
    detail_files = sorted(glob.glob(os.path.join(args.results_dir, "*_details.csv")))
    all_details = []
    for df_f in detail_files:
        if os.path.basename(df_f) not in ["similarity_threshold_details.csv"]:
            df_item = pd.read_csv(df_f)
            all_details.append(df_item)

    if all_details:
        df_all_details = pd.concat(all_details, ignore_index=True)
        df_all_details.to_csv(args.output_details_csv, index=False)
        logger.info(f"Details gespeichert unter: {args.output_details_csv}")

    # 4. Wissenschaftliche Plots generieren
    # Plot A: Datensatz-Funnel & Token-Volumen
    if not df_corpus_stats.empty:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color = '#2b5c8f'
        ax1.set_xlabel('Minimaler Ähnlichkeits-Schwellenwert ($s_{min}$)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Verbleibende Artikelpaare ($N$)', color=color, fontsize=12, fontweight='bold')
        ax1.plot(df_corpus_stats['min_sim'], df_corpus_stats['pairs'], marker='o', color=color, linewidth=2.5, label='Artikelpaare')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = '#d95f02'
        ax2.set_ylabel('Erhalt des Korpus (%)', color=color, fontsize=12, fontweight='bold')
        ax2.plot(df_corpus_stats['min_sim'], df_corpus_stats['retention_pct'], marker='s', color=color, linestyle='--', linewidth=2, label='Retention (%)')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Einfluss des Ähnlichkeits-Schwellenwerts auf das Datenvolumen', fontsize=14, fontweight='bold', pad=15)
        fig.tight_layout()
        plot_path = os.path.join(args.plots_dir, "data_retention_and_tokens.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot gespeichert: {plot_path}")

    # Plot B: MixUp Regressor Trade-off (In-Domain MSE vs. OOD AUC)
    df_mixup = df_metrics[df_metrics.get("model_type", "") == "MixUp Regressor"] if "model_type" in df_metrics.columns else pd.DataFrame()
    if not df_mixup.empty and len(df_mixup) >= 2:
        df_mixup = df_mixup.sort_values(by="min_sim")
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        c1 = '#1b9e77'
        ax1.set_xlabel('Minimaler Ähnlichkeits-Schwellenwert ($s_{min}$)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('In-Domain Test MSE (niedriger = besser)', color=c1, fontsize=12, fontweight='bold')
        ax1.plot(df_mixup['min_sim'], df_mixup['in_domain_test_mse'], marker='o', color=c1, linewidth=2.5, label='In-Domain MSE')
        ax1.tick_params(axis='y', labelcolor=c1)

        ax2 = ax1.twinx()
        c2 = '#7570b3'
        ax2.set_ylabel('OOD Lebenshilfe Separation AUC (höher = besser)', color=c2, fontsize=12, fontweight='bold')
        if 'ood_separation_auc' in df_mixup.columns:
            ax2.plot(df_mixup['min_sim'], df_mixup['ood_separation_auc'], marker='^', color=c2, linewidth=2.5, linestyle='--', label='OOD AUC')
        ax2.tick_params(axis='y', labelcolor=c2)

        plt.title('MixUp Regressor: In-Domain Güte vs. Out-of-Domain Generalisierung', fontsize=14, fontweight='bold', pad=15)
        fig.tight_layout()
        plot_path = os.path.join(args.plots_dir, "mixup_tradeoff_in_vs_ood.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot gespeichert: {plot_path}")

    # Plot C: SFT Translation Model (Simplicity vs. Semantik vs. Composite Reward)
    df_sft = df_metrics[df_metrics.get("model_type", "") == "SFT mBART-50 LoRA"] if "model_type" in df_metrics.columns else pd.DataFrame()
    if not df_sft.empty and len(df_sft) >= 2:
        df_sft = df_sft.sort_values(by="min_sim")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(df_sft['min_sim'], df_sft['r_style_mean'], marker='o', color='#e7298a', linewidth=2.5, label='Stilistische Einfachheit ($R_{style}$)')
        ax.plot(df_sft['min_sim'], df_sft['r_sem_as_mean'], marker='s', color='#1f78b4', linewidth=2.5, label='Semantischer Erhalt ($R_{sem}$)')
        ax.plot(df_sft['min_sim'], df_sft['composite_reward_mean'], marker='D', color='#33a02c', linewidth=2.5, linestyle='--', label='Composite Reward (0.5/0.5)')
        
        ax.set_xlabel('Minimaler Ähnlichkeits-Schwellenwert ($s_{min}$)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score / Reward [0.0 - 1.0]', fontsize=12, fontweight='bold')
        ax.legend(loc='best', frameon=True)
        ax.set_title('SFT Übersetzungsmodell: Stil, Semantik & Composite Reward', fontsize=14, fontweight='bold', pad=15)
        
        fig.tight_layout()
        plot_path = os.path.join(args.plots_dir, "sft_tradeoff_style_vs_sem.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot gespeichert: {plot_path}")

    # 5. LaTeX Tabellen Exportieren
    latex_out_path = os.path.join(args.results_dir, "latex_tables_similarity_ablation.tex")
    with open(latex_out_path, "w", encoding="utf-8") as f:
        f.write("% ==============================================================================\n")
        f.write("% Auto-generierte LaTeX Tabellen: Similarity Threshold Experiment\n")
        f.write("% ==============================================================================\n\n")

        # Tabelle 1: Korpus Übersicht
        f.write("% Tabelle: Korpusgrößen nach Schwellenwert\n")
        f.write("\\begin{table}[htbp]\n\\centering\\small\n")
        f.write("\\caption{Übersicht der Datenretention nach Ähnlichkeits-Schwellenwerten ($s_{max} = 0{,}98$).}\n")
        f.write("\\label{tab:similarity_threshold_corpus}\n")
        f.write("\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n")
        f.write("\\textbf{Filterbereich} & \\textbf{Paare} & \\textbf{Retention} & \\textbf{AS Tokens} & \\textbf{LS Tokens} & \\textbf{Token-Ratio} \\\\\n\\midrule\n")
        for _, r in df_corpus_stats.iterrows():
            f.write(f"${r['min_sim']:.2f} \\le \\text{{sim}} \\le 0.98$ & {int(r['pairs'])} & {r['retention_pct']:.1f}\\,\\% & {int(r['as_tokens']):,} & {int(r['ls_tokens']):,} & {r['token_ratio']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    logger.info(f"LaTeX Tabellen gespeichert unter: {latex_out_path}")
    logger.info("=== Konsolidierung erfolgreich abgeschlossen! ===")


if __name__ == "__main__":
    main()
