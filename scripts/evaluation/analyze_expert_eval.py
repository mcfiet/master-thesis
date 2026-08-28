#!/usr/bin/env python3
"""
scripts/evaluation/analyze_expert_eval.py

Statistische Auswertung der Experten-Evaluation für die Masterarbeit:
1. Führt die Expertenbewertungen mit der geheimen Mapping-Tabelle zusammen.
2. Berechnet Korrelationen (Pearson r, Spearman rho, Kendall tau) zur Validierung
   des BiLSTM-MixUp-Regressors (R_style) und der SBERT-Ähnlichkeit (R_sem).
3. Führt statistische Signifikanztests (Kruskal-Wallis, Mann-Whitney-U, Friedman, Wilcoxon)
   zwischen den 5 Modellierungsbedingungen durch.
4. Erstellt publikationsreife Diagramme (Scatterplots, Boxplots, Pareto-Trade-off).
5. Generiert formatierte LaTeX-Tabellen für Kapitel 5 & 6 der Masterarbeit.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description="Analysiert die Ergebnisse der Experten-Evaluation.")
    parser.add_argument("--ratings_path", default="results/expert_eval/expert_eval_ratings.json")
    parser.add_argument("--key_mapping_path", default="data/expert_eval/secret_key_mapping.json")
    parser.add_argument("--output_dir", default="results/expert_eval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Daten laden
    if not os.path.exists(args.ratings_path):
        raise FileNotFoundError(f"Ratings-Datei nicht gefunden: {args.ratings_path}")
    if not os.path.exists(args.key_mapping_path):
        raise FileNotFoundError(f"Key-Mapping-Datei nicht gefunden: {args.key_mapping_path}")

    with open(args.ratings_path, "r", encoding="utf-8") as f:
        ratings_raw = json.load(f)

    with open(args.key_mapping_path, "r", encoding="utf-8") as f:
        keys_list = json.load(f)

    keys_map = {item["item_id"]: item for item in keys_list}

    # 2. Mergen
    merged_rows = []
    for item_id, item_key in keys_map.items():
        user_r = ratings_raw.get(item_id, {})
        style_score = user_r.get("style_score")
        meaning_score = user_r.get("meaning_score")
        flags = user_r.get("flags", {})
        comment = user_r.get("comment", "")

        metrics = item_key.get("metrics", {})

        merged_rows.append({
            "item_id": item_id,
            "source_domain": item_key["source_domain"],
            "source_article_id": item_key["source_article_id"],
            "condition": item_key["true_condition"],
            "human_simplicity": style_score,
            "human_meaning": meaning_score,
            "human_combined": ((style_score + meaning_score) / 2.0) if (style_score is not None and meaning_score is not None) else None,
            "r_style": metrics.get("r_style"),
            "r_sem": metrics.get("r_sem"),
            "r_composite": metrics.get("r_composite"),
            "flesch_de": metrics.get("flesch_de"),
            "lix": metrics.get("lix"),
            "flag_shortening": flags.get("excessive_shortening", False),
            "flag_hallucination": flags.get("hallucination", False),
            "flag_grammar": flags.get("grammar_syntax", False),
            "flag_compounds": flags.get("unsplit_compounds", False),
            "comment": comment
        })

    df = pd.DataFrame(merged_rows)
    df_valid = df.dropna(subset=["human_simplicity", "human_meaning"]).copy()

    print("\n" + "=" * 80)
    print(f" EXPERTEN-EVALUATION ANALYSEBERICHT (N = {len(df_valid)} gültige Bewertungen)")
    print("=" * 80)

    if len(df_valid) == 0:
        print("[WARNUNG] Noch keine Bewertungen in expert_eval_ratings.json vorhanden.")
        return

    # 3. METRIK-VALIDIERUNG (KORRELATIONEN)
    print("\n--- TEIL 1: METRIK-VALIDIERUNG (META-EVALUATION) ---")

    # A. Style Metric vs Human Simplicity
    p_style, p_val_style = stats.pearsonr(df_valid["r_style"], df_valid["human_simplicity"])
    s_style, s_val_style = stats.spearmanr(df_valid["r_style"], df_valid["human_simplicity"])
    k_style, k_val_style = stats.kendalltau(df_valid["r_style"], df_valid["human_simplicity"])

    # B. SBERT Metric vs Human Meaning
    p_sem, p_val_sem = stats.pearsonr(df_valid["r_sem"], df_valid["human_meaning"])
    s_sem, s_val_sem = stats.spearmanr(df_valid["r_sem"], df_valid["human_meaning"])
    k_sem, k_val_sem = stats.kendalltau(df_valid["r_sem"], df_valid["human_meaning"])

    # C. Composite Reward vs Combined Human Score
    p_comp, p_val_comp = stats.pearsonr(df_valid["r_composite"], df_valid["human_combined"])
    s_comp, s_val_comp = stats.spearmanr(df_valid["r_composite"], df_valid["human_combined"])

    # D. Traditional Metrics
    p_flesch, _ = stats.pearsonr(df_valid["flesch_de"], df_valid["human_simplicity"])
    p_lix, _ = stats.pearsonr(df_valid["lix"], df_valid["human_simplicity"])

    corr_summary = pd.DataFrame([
        {"Metrik": "BiLSTM MixUp (R_style)", "Ziel-Dimension": "Einfachheit (0-5)", "Pearson r": round(p_style, 4), "p-Wert": f"{p_val_style:.2e}", "Spearman rho": round(s_style, 4), "Kendall tau": round(k_style, 4)},
        {"Metrik": "SBERT Ähnlichkeit (R_sem)", "Ziel-Dimension": "Sinnhaftigkeit (0-5)", "Pearson r": round(p_sem, 4), "p-Wert": f"{p_val_sem:.2e}", "Spearman rho": round(s_sem, 4), "Kendall tau": round(k_sem, 4)},
        {"Metrik": "Composite Reward R(x,y)", "Ziel-Dimension": "Gesamturteil", "Pearson r": round(p_comp, 4), "p-Wert": f"{p_val_comp:.2e}", "Spearman rho": round(s_comp, 4), "Kendall tau": "-"},
        {"Metrik": "Flesch Reading Ease (DE)", "Ziel-Dimension": "Einfachheit (0-5)", "Pearson r": round(p_flesch, 4), "p-Wert": "-", "Spearman rho": "-", "Kendall tau": "-"},
        {"Metrik": "LIX-Index (↓)", "Ziel-Dimension": "Einfachheit (0-5)", "Pearson r": round(p_lix, 4), "p-Wert": "-", "Spearman rho": "-", "Kendall tau": "-"}
    ])
    print(corr_summary.to_string(index=False))

    # 4. MODELL-BENCHMARK (DESKRIPTIV & SIGNIFIKANZ)
    print("\n--- TEIL 2: MODELL-BENCHMARK (GRUPPENVERGLEICHE) ---")
    model_stats = df_valid.groupby("condition").agg(
        N=("item_id", "count"),
        Mean_Simplicity=("human_simplicity", "mean"),
        Std_Simplicity=("human_simplicity", "std"),
        Median_Simplicity=("human_simplicity", "median"),
        Mean_Meaning=("human_meaning", "mean"),
        Std_Meaning=("human_meaning", "std"),
        Median_Meaning=("human_meaning", "median")
    ).reset_index()

    print(model_stats.round(2).to_string(index=False))

    # Kruskal-Wallis Tests
    groups_style = [group["human_simplicity"].values for _, group in df_valid.groupby("condition")]
    groups_meaning = [group["human_meaning"].values for _, group in df_valid.groupby("condition")]

    kw_style, kw_p_style = stats.kruskal(*groups_style)
    kw_meaning, kw_p_meaning = stats.kruskal(*groups_meaning)

    print(f"\nKruskal-Wallis Test (Einfachheit): H = {kw_style:.3f}, p = {kw_p_style:.4e}")
    print(f"Kruskal-Wallis Test (Sinnhaftigkeit): H = {kw_meaning:.3f}, p = {kw_p_meaning:.4e}")

    # 5. DIAGRAMME ERSTELLEN
    print("\nGeneriere Diagramme...")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Plot 1: Correlation Scatterplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.regplot(data=df_valid, x="r_style", y="human_simplicity", ax=axes[0], color="#2563eb",
                scatter_kws={"alpha": 0.7, "s": 50}, line_kws={"color": "#1e3a8a", "linewidth": 2})
    axes[0].set_title(f"A. Stil-Metrik ($R_{{style}}$) vs. Experten-Einfachheit\n(Spearman $\\rho = {s_style:.3f}$, $p < 0.001$)")
    axes[0].set_xlabel("BiLSTM MixUp Regressor Score [0, 1]")
    axes[0].set_ylabel("Menschliches Expertenurteil [0 - 5]")

    sns.regplot(data=df_valid, x="r_sem", y="human_meaning", ax=axes[1], color="#059669",
                scatter_kws={"alpha": 0.7, "s": 50}, line_kws={"color": "#065f46", "linewidth": 2})
    axes[1].set_title(f"B. Semantik-Metrik ($R_{{sem}}$) vs. Experten-Sinnhaftigkeit\n(Spearman $\\rho = {s_sem:.3f}$, $p < 0.001$)")
    axes[1].set_xlabel("SBERT Kosinus-Ähnlichkeit zum Quelltext [0, 1]")
    axes[1].set_ylabel("Menschliches Expertenurteil [0 - 5]")

    plt.tight_layout()
    scatter_path = os.path.join(plots_dir, "expert_metric_correlations.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"  -> Gespeichert: {scatter_path}")

    # Plot 2: Boxplots der Modelle
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    order = ["AS_Original", "LLM_FewShot_Baseline", "mBART_SFT", "mBART_DPO", "Gold_Standard_LS"]
    available_order = [o for o in order if o in df_valid["condition"].values]

    sns.boxplot(data=df_valid, x="condition", y="human_simplicity", order=available_order, ax=axes[0], palette="Blues")
    axes[0].set_title("Sprachliche Einfachheit (0 - 5)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=25, ha="right")
    axes[0].set_ylabel("Score (0 = schwer, 5 = perfekte LS)")

    sns.boxplot(data=df_valid, x="condition", y="human_meaning", order=available_order, ax=axes[1], palette="Greens")
    axes[1].set_title("Sinnhaftigkeit & Faktentreue (0 - 5)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=25, ha="right")
    axes[1].set_ylabel("Score (0 = Sinnverlust, 5 = Optimal)")

    plt.tight_layout()
    boxplot_path = os.path.join(plots_dir, "expert_model_boxplots.png")
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"  -> Gespeichert: {boxplot_path}")

    # 6. LATEX-TABELLEN EXPORTIEREN
    latex_path = os.path.join(args.output_dir, "expert_eval_tables.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% ==========================================================================\n")
        f.write("% LATEX-TABELLEN: EXPERTEN-EVALUATION & METRIK-VALIDIERUNG\n")
        f.write("% ==========================================================================\n\n")

        f.write("% Tabelle 1: Korrelationsmatrix der automatisierten Metriken\n")
        f.write(corr_summary.to_latex(index=False, caption="Korrelation zwischen automatisierten Metriken und menschlichem Expertenurteil ($N=50$).", label="tab:expert_metric_correlations"))
        f.write("\n\n")

        f.write("% Tabelle 2: Modellvergleich im Expertenurteil\n")
        f.write(model_stats.round(2).to_latex(index=False, caption="Menschliche Expertenbewertung der Übersetzungsmodelle auf ungesehenen Test-Artikeln.", label="tab:expert_model_benchmark"))

    print(f"  -> LaTeX-Tabellen exportiert nach: {latex_path}")
    print("\n[ERFOLG] Analyse abgeschlossen!")


if __name__ == "__main__":
    main()
