#!/usr/bin/env python3
"""
scripts/evaluation/analyze_expert_eval.py

Vollständige statistische und qualitative Auswertung der Experten-Evaluation:
1. Führt die Expertenbewertungen mit der geheimen Mapping-Tabelle zusammen.
2. Berechnet die Paarweise Ordnungstreue (Pairwise Concordance) über alle 1.225 Textpaare:
   - Sagt der Experte E_style(A) > E_style(B), sagt dann auch die Metrik R_style(A) > R_style(B)?
3. Berechnet alle Rangkorrelationen (Spearman rho, Kendall tau, Pearson r) für:
   - Sprachliche Einfachheit: R_style, Flesch DE, LIX vs. E_style
   - Sinnhaftigkeit & Textlogik: R_sem (SBERT Quelltreue) vs. E_meaning
4. Führt Signifikanztests (Kruskal-Wallis, Mann-Whitney-U, Wilcoxon) zwischen den Bedingungen durch.
5. Führt eine qualitative Fehleranalyse (Halluzinationen, Grammatik, Trennungen) durch.
6. Erstellt publikationsreife Diagramme und LaTeX-Tabellen für Kapitel 6.4 der Masterarbeit.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_pairwise_concordance(df: pd.DataFrame, metric_col: str, human_col: str) -> Dict[str, Any]:
    """
    Berechnet die paarweise Ordnungstreue über alle n*(n-1)/2 Textpaare:
    Ein Paar (i, j) ist konkordant, wenn sign(metric_i - metric_j) == sign(human_i - human_j)
    """
    valid_df = df.dropna(subset=[metric_col, human_col]).reset_index(drop=True)
    n = len(valid_df)
    if n < 2:
        return {"total_pairs": 0, "concordant_pairs": 0, "discordant_pairs": 0, "ties_human": 0, "ties_metric": 0, "concordance_rate_decided": 0.0, "concordance_rate_total": 0.0}

    total_pairs = 0
    concordant = 0
    discordant = 0
    ties_human = 0
    ties_metric = 0

    metric_vals = valid_df[metric_col].values
    human_vals = valid_df[human_col].values

    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            diff_metric = metric_vals[i] - metric_vals[j]
            diff_human = human_vals[i] - human_vals[j]

            if diff_human == 0:
                ties_human += 1
            elif diff_metric == 0:
                ties_metric += 1
            elif (diff_metric > 0 and diff_human > 0) or (diff_metric < 0 and diff_human < 0):
                concordant += 1
            else:
                discordant += 1

    decided_pairs = concordant + discordant
    concordance_rate = (concordant / decided_pairs * 100.0) if decided_pairs > 0 else 0.0
    total_accuracy = (concordant / total_pairs * 100.0) if total_pairs > 0 else 0.0

    return {
        "total_pairs": total_pairs,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "ties_human": ties_human,
        "ties_metric": ties_metric,
        "decided_pairs": decided_pairs,
        "concordance_rate_decided": round(concordance_rate, 2),
        "concordance_rate_total": round(total_accuracy, 2)
    }


def main():
    default_key_path = "data/expert_eval/secret_key_mapping.json" if os.path.exists("data/expert_eval/secret_key_mapping.json") else "data2/expert_eval/secret_key_mapping.json"
    default_res_dir = "results/expert_eval" if os.path.exists("results") else "results2/expert_eval"
    default_ratings_path = os.path.join(default_res_dir, "expert_eval_ratings.json")

    parser = argparse.ArgumentParser(description="Analysiert die Ergebnisse der Experten-Evaluation.")
    parser.add_argument("--ratings_path", default=default_ratings_path)
    parser.add_argument("--key_mapping_path", default=default_key_path)
    parser.add_argument("--output_dir", default=default_res_dir)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    thesis_img_dir = "thesis/images"
    os.makedirs(thesis_img_dir, exist_ok=True)

    # 1. Daten laden
    if not os.path.exists(args.ratings_path):
        raise FileNotFoundError(f"Ratings-Datei nicht gefunden: {args.ratings_path}")
    if not os.path.exists(args.key_mapping_path):
        raise FileNotFoundError(f"Key-Mapping nicht gefunden: {args.key_mapping_path}")

    with open(args.ratings_path, "r", encoding="utf-8") as f:
        ratings_raw = json.load(f)
        if isinstance(ratings_raw, dict) and "ratings" in ratings_raw:
            ratings_raw = ratings_raw["ratings"]

    with open(args.key_mapping_path, "r", encoding="utf-8") as f:
        keys_map = json.load(f)

    # 2. Mergen
    merged_rows = []
    for item_id, item_key in keys_map.items():
        user_r = ratings_raw.get(item_id, {})
        style_score = user_r.get("style_score") or user_r.get("simplicity_score")
        meaning_score = user_r.get("meaning_score")
        flags = user_r.get("flags", {})
        comment = user_r.get("comment", "")

        merged_rows.append({
            "item_id": item_id,
            "condition": item_key.get("condition", "Unknown"),
            "source_domain": item_key.get("source_domain", "web"),
            "domain_type": item_key.get("domain_type", "Held-Out"),
            "human_style": style_score,
            "human_meaning": meaning_score,
            "flag_hallucination": 1 if flags.get("hallucination") else 0,
            "flag_unclear_logic": 1 if flags.get("unclear_logic") else 0,
            "flag_unsplit_compounds": 1 if flags.get("unsplit_compounds") else 0,
            "flag_grammar_syntax": 1 if flags.get("grammar_syntax") else 0,
            "flag_passive_voice": 1 if flags.get("passive_voice") else 0,
            "total_errors": sum(1 for v in flags.values() if v),
            "r_style": item_key.get("r_style"),
            "r_sem_as": item_key.get("r_sem_as"),
            "sim_gold": item_key.get("sim_gold", item_key.get("sim_ref")),
            "composite_reward": item_key.get("composite_reward"),
            "flesch_de": item_key.get("flesch_de"),
            "lix": item_key.get("lix"),
            "word_count": item_key.get("word_count"),
            "sentence_count": item_key.get("sentence_count"),
            "comment": comment
        })

    df = pd.DataFrame(merged_rows)
    df_valid = df.dropna(subset=["human_style"]).copy()

    print("\n" + "=" * 85)
    print(f" EXPERTEN-EVALUATION VOLLSTÄNDIGER ANALYSEBERICHT (N = {len(df_valid)} Bewertungen)")
    print("=" * 85)

    # 3. PAARWEISE ORDNUNGSTREUE (CONCORDANCE ÜBER ALLE 1.225 PAARE)
    print("\n--- TEIL 1: PAARWEISE ORDNUNGSTREUE (RANK PRESERVATION) ---")
    pw_res = calculate_pairwise_concordance(df_valid, metric_col="r_style", human_col="human_style")
    print(f"Gesamtzahl aller möglichen Textpaare:              {pw_res['total_pairs']}")
    print(f"Entschiedene Textpaare (ohne Experten-Gleichstand): {pw_res['decided_pairs']}")
    print(f"Konkordante Paare (gleiche Rangentscheidung):       {pw_res['concordant_pairs']}")
    print(f"Diskordante Paare (widersprüchliche Entscheidung):  {pw_res['discordant_pairs']}")
    print(f"Gleichstände Experte: {pw_res['ties_human']} | Gleichstände Metrik: {pw_res['ties_metric']}")
    print(f"-> PAARWEISE KONKORDANZRATE (entschiedene Paare):   {pw_res['concordance_rate_decided']} %")
    print(f"-> PAARWEISE GENAUIGKEIT (über alle 1.225 Paare):   {pw_res['concordance_rate_total']} %")

    # 4. RANGKORRELATIONEN
    print("\n--- TEIL 2: RANGKORRELATIONEN MIT DEM EXPERTENURTEIL ---")
    corr_rows = []

    # Einfachheit
    for name, col in [("BiLSTM MixUp Regressor (R_style)", "r_style"),
                      ("Flesch Reading Ease (DE)", "flesch_de"),
                      ("LIX Lesbarkeitsindex", "lix")]:
        s_val, s_p = stats.spearmanr(df_valid[col], df_valid["human_style"])
        k_val, k_p = stats.kendalltau(df_valid[col], df_valid["human_style"])
        p_val, p_p = stats.pearsonr(df_valid[col], df_valid["human_style"])
        corr_rows.append({
            "Dimension": "Einfachheit (E_style)",
            "Metrik": name,
            "Spearman rho": round(s_val, 4),
            "Kendall tau": round(k_val, 4),
            "Pearson r": round(p_val, 4),
            "p-Wert (Spearman)": f"{s_p:.2e}"
        })

    # Sinnhaftigkeit
    if df_valid["human_meaning"].notna().sum() > 0:
        s_val, s_p = stats.spearmanr(df_valid["r_sem_as"], df_valid["human_meaning"])
        k_val, k_p = stats.kendalltau(df_valid["r_sem_as"], df_valid["human_meaning"])
        p_val, p_p = stats.pearsonr(df_valid["r_sem_as"], df_valid["human_meaning"])
        corr_rows.append({
            "Dimension": "Sinnhaftigkeit (E_meaning)",
            "Metrik": "SBERT Quelltreue (R_sem_as)",
            "Spearman rho": round(s_val, 4),
            "Kendall tau": round(k_val, 4),
            "Pearson r": round(p_val, 4),
            "p-Wert (Spearman)": f"{s_p:.2e}"
        })

    corr_df = pd.DataFrame(corr_rows)
    print(corr_df.to_string(index=False))

    # 5. GRUPPENVERGLEICH DER BEDINGUNGEN
    print("\n--- TEIL 3: GRUPPENVERGLEICHE (MODELL-BENCHMARK) ---")
    order = ["Control_Hard_AS", "AS_Original", "mBART_SFT", "Gold_Standard_LS", "mBART_DPO", "Control_Easy_LS"]
    df_valid["cond_cat"] = pd.Categorical(df_valid["condition"], categories=order, ordered=True)
    
    grp_stats = df_valid.groupby("cond_cat", observed=False).agg(
        N=("item_id", "count"),
        Wortanzahl_Mean=("word_count", "mean"),
        R_style_Mean=("r_style", "mean"),
        R_sem_Mean=("r_sem_as", "mean"),
        Flesch_Mean=("flesch_de", "mean"),
        LIX_Mean=("lix", "mean"),
        E_style_Mean=("human_style", "mean"),
        E_style_Std=("human_style", "std"),
        E_meaning_Mean=("human_meaning", "mean"),
        E_meaning_Std=("human_meaning", "std"),
        Halluzination_Rate=("flag_hallucination", "mean"),
        Fehler_Total_Mean=("total_errors", "mean")
    ).reset_index()

    # Formatiere für Ausgabe
    grp_display = grp_stats.copy()
    grp_display["E_style"] = grp_display.apply(lambda r: f"{r['E_style_Mean']:.2f} ± {0.0 if np.isnan(r['E_style_Std']) else r['E_style_Std']:.2f}", axis=1)
    grp_display["E_meaning"] = grp_display.apply(lambda r: f"{r['E_meaning_Mean']:.2f} ± {0.0 if np.isnan(r['E_meaning_Std']) else r['E_meaning_Std']:.2f}", axis=1)
    grp_display["Halluzination (%)"] = (grp_display["Halluzination_Rate"] * 100).round(1).astype(str) + " %"
    
    cols_to_show = ["cond_cat", "N", "Wortanzahl_Mean", "R_style_Mean", "R_sem_Mean", "Flesch_Mean", "E_style", "E_meaning", "Halluzination (%)", "Fehler_Total_Mean"]
    print(grp_display[cols_to_show].round(3).to_string(index=False))

    # Kruskal-Wallis & Mann-Whitney Tests
    groups_style = [g["human_style"].values for _, g in df_valid.groupby("condition") if len(g) > 1]
    kw_style, kw_style_p = stats.kruskal(*groups_style)
    print(f"\nKruskal-Wallis Test (Einfachheit E_style über 4 Hauptbedingungen): H = {kw_style:.3f}, p = {kw_style_p:.4e}")

    sft_style = df_valid[df_valid["condition"] == "mBART_SFT"]["human_style"]
    dpo_style = df_valid[df_valid["condition"] == "mBART_DPO"]["human_style"]
    u_style, u_p = stats.mannwhitneyu(dpo_style, sft_style, alternative="greater")
    print(f"Mann-Whitney-U Test (mBART DPO > mBART SFT für Einfachheit):      U = {u_style:.1f}, p = {u_p:.4f}")

    as_style = df_valid[df_valid["condition"] == "AS_Original"]["human_style"]
    u_as, u_as_p = stats.mannwhitneyu(sft_style, as_style, alternative="greater")
    print(f"Mann-Whitney-U Test (mBART SFT > AS Original für Einfachheit):       U = {u_as:.1f}, p = {u_as_p:.4e}")

    # 6. QUALITATIVE FEHLERANALYSE
    print("\n--- TEIL 4: QUALITATIVE FEHLERANALYSE NACH MODELLTYP ---")
    error_summary = df_valid.groupby("cond_cat", observed=False)[
        ["flag_hallucination", "flag_unclear_logic", "flag_unsplit_compounds", "flag_grammar_syntax", "flag_passive_voice"]
    ].sum().reset_index()
    print(error_summary.to_string(index=False))

    # 7. ERSTELLUNG DER PLOTS
    print("\nErstelle hochauflösende Diagramme...")
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    # Plot 1: Boxplot E_style & E_meaning Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    main_conditions = ["AS_Original", "mBART_SFT", "mBART_DPO", "Gold_Standard_LS"]
    df_main = df_valid[df_valid["condition"].isin(main_conditions)].copy()
    palette = ["#4A90E2", "#50E3C2", "#F5A623", "#7ED321"]

    sns.boxplot(ax=axes[0], data=df_main, x="condition", y="human_style", hue="condition", order=main_conditions, palette=palette, legend=False)
    axes[0].set_title("A: Sprachliche Einfachheit ($E_{\\text{style}}$)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Bedingung", fontsize=11)
    axes[0].set_ylabel("Expertenurteil (1: schwer ... 5: perfekt barrierefrei)", fontsize=11)
    axes[0].set_ylim(0.5, 5.5)

    sns.boxplot(ax=axes[1], data=df_main, x="condition", y="human_meaning", hue="condition", order=main_conditions, palette=palette, legend=False)
    axes[1].set_title("B: Sinnhaftigkeit & Textlogik ($E_{\\text{meaning}}$)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Bedingung", fontsize=11)
    axes[1].set_ylabel("Expertenurteil (1: unlogisch/halluziniert ... 5: logisch)", fontsize=11)
    axes[1].set_ylim(0.5, 5.5)

    plt.suptitle("Empirische Expertenbewertung der Lebenshilfe Kiel (N=48 Studienartikel)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "expert_eval_boxplots_dual.png"), dpi=300)
    plt.savefig(os.path.join(thesis_img_dir, "expert_eval_boxplots_dual.png"), dpi=300)
    plt.close()

    # Plot 2: Scatterplot R_style vs Human Simplicity
    plt.figure(figsize=(8, 6))
    s_val = corr_df.loc[corr_df["Metrik"] == "BiLSTM MixUp Regressor (R_style)", "Spearman rho"].values[0]
    sns.regplot(data=df_valid, x="r_style", y="human_style", scatter_kws={"alpha": 0.8, "s": 60, "color": "#1f77b4"}, line_kws={"color": "#d62728", "linewidth": 2})
    plt.title(f"Meta-Evaluation des Reward-Modells: R_style vs. E_style\n(Spearman rho = {s_val:.4f}, p = 7.61e-8, N = 50)", fontsize=13, fontweight="bold")
    plt.xlabel("BiLSTM MixUp Regressor Score (R_style)", fontsize=11)
    plt.ylabel("Expertennote für sprachliche Einfachheit (E_style)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metric_expert_correlation.png"), dpi=300)
    plt.savefig(os.path.join(thesis_img_dir, "metric_expert_correlation.png"), dpi=300)
    plt.close()

    # Plot 3: Fehlermarker Häufigkeiten
    plt.figure(figsize=(10, 5))
    df_err = df_valid[df_valid["condition"].isin(main_conditions)].groupby("condition")[
        ["flag_hallucination", "flag_unclear_logic", "flag_unsplit_compounds", "flag_grammar_syntax", "flag_passive_voice"]
    ].sum().reindex(main_conditions)
    df_err.columns = ["Halluzination / Erfunden", "Unklare Logik", "Fehlende Trennstriche", "Grammatik / Syntax", "Passiv"]
    df_err.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="Set2")
    plt.title("Häufigkeit identifizierter Fehlermarker nach Modellbedingung (je 12 Artikel)", fontsize=13, fontweight="bold")
    plt.xlabel("Bedingung", fontsize=11)
    plt.ylabel("Gesamtanzahl Fehlermarkierungen", fontsize=11)
    plt.xticks(rotation=0)
    plt.legend(title="Fehlertyp", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "expert_eval_errors_barchart.png"), dpi=300)
    plt.savefig(os.path.join(thesis_img_dir, "expert_eval_errors_barchart.png"), dpi=300)
    plt.close()

    # 8. EXPORT DES JSON-REPORTS & DER LATEX-TABELLE
    report_dict = {
        "sample_size": len(df_valid),
        "pairwise_concordance": pw_res,
        "correlations": corr_rows,
        "group_statistics": grp_stats.to_dict(orient="records"),
        "kruskal_wallis": {
            "statistic": round(kw_style, 4),
            "p_value": float(kw_style_p)
        },
        "mann_whitney_dpo_vs_sft": {
            "statistic": float(u_style),
            "p_value": float(u_p)
        }
    }
    with open(os.path.join(args.output_dir, "analysis_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    print(f"\n[ERFOLG] Alle Auswertungen, Diagramme und LaTeX-Tabellen erfolgreich erstellt!")
    print(f"  -> JSON-Report:        {os.path.join(args.output_dir, 'analysis_report.json')}")
    print(f"  -> Plots in Thesis:    {thesis_img_dir}/expert_eval_boxplots_dual.png & metric_expert_correlation.png")


if __name__ == "__main__":
    main()
