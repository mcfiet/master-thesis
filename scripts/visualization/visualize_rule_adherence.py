#!/usr/bin/env python3
"""
Visualizes quantitative rule-adherence analysis for Leichte Sprache:
- Source comparison across the 12 corpus sources (Percentages + Absolute Side-by-Side AS vs. LS)
- Comprehensive Model Dashboard across all 9-14 rule metrics (AS vs. SFT vs. Lebenshilfe Gold)
- Multi-dimensional Radar chart
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300

OUTPUT_DIR = "research/img/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_corpus_source_side_by_side(corpus_csv: str):
    """Plots absolute side-by-side barplots (AS vs. LS) for each source."""
    if not os.path.exists(corpus_csv):
        return
    df = pd.read_csv(corpus_csv)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. Satzlänge (Wörter pro Satz)
    df_sent = df.groupby("source")[["as_avg_sent_len", "ls_avg_sent_len"]].mean().reset_index()
    df_sent_melted = df_sent.melt(id_vars="source", var_name="Texttyp", value_name="Satzlänge (Wörter)")
    df_sent_melted["Texttyp"] = df_sent_melted["Texttyp"].map({"as_avg_sent_len": "Ausgangstext (AS)", "ls_avg_sent_len": "Leichte Sprache (LS)"})

    sns.barplot(data=df_sent_melted, x="source", y="Satzlänge (Wörter)", hue="Texttyp", ax=axes[0, 0], palette=["#4c72b0", "#55a868"])
    axes[0, 0].set_title("1. Satzlänge im direkten Vergleich: AS vs. LS (Wörter pro Satz)", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].axhline(8.0, color='darkgreen', linestyle=':', label="Richtwert LS (≤ 8–10 Wörter)")
    axes[0, 0].legend()

    # 2. Genitiv-Quote
    df_gen = df.groupby("source")[["as_genitive_ratio", "ls_genitive_ratio"]].mean().reset_index()
    df_gen_melted = df_gen.melt(id_vars="source", var_name="Texttyp", value_name="Genitiv-Quote")
    df_gen_melted["Texttyp"] = df_gen_melted["Texttyp"].map({"as_genitive_ratio": "Ausgangstext (AS)", "ls_genitive_ratio": "Leichte Sprache (LS)"})

    sns.barplot(data=df_gen_melted, x="source", y="Genitiv-Quote", hue="Texttyp", ax=axes[0, 1], palette=["#4c72b0", "#55a868"])
    axes[0, 1].set_title("2. Genitiv-Quote im direkten Vergleich: AS vs. LS (Genitive/Nomen)", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend()

    # 3. Passiv pro Satz
    df_pass = df.groupby("source")[["as_passive_ratio", "ls_passive_ratio"]].mean().reset_index()
    df_pass_melted = df_pass.melt(id_vars="source", var_name="Texttyp", value_name="Passiv pro Satz")
    df_pass_melted["Texttyp"] = df_pass_melted["Texttyp"].map({"as_passive_ratio": "Ausgangstext (AS)", "ls_passive_ratio": "Leichte Sprache (LS)"})

    sns.barplot(data=df_pass_melted, x="source", y="Passiv pro Satz", hue="Texttyp", ax=axes[1, 0], palette=["#4c72b0", "#55a868"])
    axes[1, 0].set_title("3. Passiv-Dichte im direkten Vergleich: AS vs. LS (Passiv/Satz)", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Quelle")
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend()

    # 4. Wiener Sachtextformel (WSTF)
    df_wstf = df.groupby("source")[["as_wstf_score", "ls_wstf_score"]].mean().reset_index()
    df_wstf_melted = df_wstf.melt(id_vars="source", var_name="Texttyp", value_name="WSTF Score")
    df_wstf_melted["Texttyp"] = df_wstf_melted["Texttyp"].map({"as_wstf_score": "Ausgangstext (AS)", "ls_wstf_score": "Leichte Sprache (LS)"})

    sns.barplot(data=df_wstf_melted, x="source", y="WSTF Score", hue="Texttyp", ax=axes[1, 1], palette=["#4c72b0", "#55a868"])
    axes[1, 1].set_title("4. Wiener Sachtextformel im direkten Vergleich (Schulstufe, Ziel ≤ 6)", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Quelle")
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(6.0, color='darkgreen', linestyle=':', label="Zielkorridor LS (≤ 6)")
    axes[1, 1].legend()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "rule_adherence_corpus_side_by_side.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_corpus_source_percentages(corpus_csv: str):
    """Plots percentage reduction barplots by source."""
    if not os.path.exists(corpus_csv):
        return
    df = pd.read_csv(corpus_csv)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    df_plot = df.groupby("source")[["red_red_sent_len_pct", "red_red_genitive_pct", "red_red_passive_pct", "ls_wstf_score"]].mean().reset_index()

    # 1. Satzlängen-Reduktion
    sns.barplot(data=df_plot.sort_values("red_red_sent_len_pct", ascending=False), 
                x="source", y="red_red_sent_len_pct", ax=axes[0, 0], hue="source", legend=False, palette="crest")
    axes[0, 0].set_title("1. Satzlängen-Kürzung von AS zu LS (%)", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Reduktion (%)")
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Genitiv-Reduktion
    sns.barplot(data=df_plot.sort_values("red_red_genitive_pct", ascending=False), 
                x="source", y="red_red_genitive_pct", ax=axes[0, 1], hue="source", legend=False, palette="Blues_r")
    axes[0, 1].set_title("2. Genitiv-Tilgung von AS zu LS (%)", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Reduktion (%)")
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Passiv-Reduktion
    sns.barplot(data=df_plot.sort_values("red_red_passive_pct", ascending=False), 
                x="source", y="red_red_passive_pct", ax=axes[1, 0], hue="source", legend=False, palette="flare")
    axes[1, 0].set_title("3. Passiv-Reduktion von AS zu LS (%)", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Quelle")
    axes[1, 0].set_ylabel("Reduktion (%)")
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. Wiener Sachtextformel LS
    sns.barplot(data=df_plot.sort_values("ls_wstf_score", ascending=True), 
                x="source", y="ls_wstf_score", ax=axes[1, 1], hue="source", legend=False, palette="magma_r")
    axes[1, 1].set_title("4. Mittlere Wiener Sachtextformel in LS (niedriger = einfacher)", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Quelle")
    axes[1, 1].set_ylabel("WSTF Score (Schulstufe)")
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(6.0, color='green', linestyle=':', label="Zielkorridor LS (≤ 6)")
    axes[1, 1].legend()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "rule_adherence_corpus_sources.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_comprehensive_model_dashboard(ladder_csv: str):
    """Plots a 3x3 comprehensive multi-metric dashboard."""
    if not os.path.exists(ladder_csv):
        return
    df = pd.read_csv(ladder_csv)

    metrics_to_plot = [
        ("avg_sent_len", "1. Satzlänge (Wörter/Satz)", "Wörter", 8.0, "Richtwert LS (≤ 8–10)"),
        ("long_sent_ratio", "2. Anteil langer Sätze (>12 Wörter)", "Anteil", None, None),
        ("subord_ratio", "3. Nebensatz-Dichte (pro Satz)", "Nebensätze/Satz", None, None),
        ("passive_ratio", "4. Passiv-Dichte (pro Satz)", "Passiv/Satz", None, None),
        ("genitive_ratio", "5. Genitiv-Quote (pro Nomen)", "Genitiv-Quote", None, None),
        ("nominal_ratio", "6. Nominalstil-Quote (Suffixe)", "Nominal-Anteil", None, None),
        ("polysyllable_ratio", "7. Polysilben-Quote (≥3 Silben)", "Polysilben-Anteil", None, None),
        ("hyphen_compound_ratio", "8. Komposita-Trennung (Bindestrich)", "Getrennte Nomen", None, None),
        ("wstf_score", "9. Wiener Sachtextformel (WSTF)", "Schulstufe", 6.0, "Zielkorridor LS (≤ 6)")
    ]

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    axes = axes.flatten()

    for idx, (key, title, ylabel, threshold, thresh_label) in enumerate(metrics_to_plot):
        as_val = df[f"as_{key}"].mean()
        gen_val = df[f"gen_{key}"].mean()
        ref_val = df[f"ref_{key}"].mean()

        data_rows = [
            {"Stufe": "1. Ausgangstext (AS)", "Wert": as_val},
            {"Stufe": "2. SFT Modell (500)", "Wert": gen_val},
            {"Stufe": "3. Lebenshilfe Gold", "Wert": ref_val}
        ]
        df_p = pd.DataFrame(data_rows)

        ax = axes[idx]
        sns.barplot(data=df_p, x="Stufe", y="Wert", ax=ax, hue="Stufe", legend=False, palette=["#4c72b0", "#c44e52", "#55a868"])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis='x', rotation=15)

        if threshold is not None:
            ax.axhline(threshold, color="darkgreen", linestyle=":", label=thresh_label)
            ax.legend(fontsize=9)

    plt.suptitle("Umfassendes Regel-Adhärenz Dashboard: AS Original vs. SFT Modell vs. Lebenshilfe Gold", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "rule_adherence_comprehensive_dashboard.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_source_radar(corpus_csv: str):
    if not os.path.exists(corpus_csv):
        return
    df = pd.read_csv(corpus_csv)
    
    key_sources = ["brandeins", "mdr", "hamburg", "koeln", "sozialpolitik", "apotheken"]
    df_sub = df[df["source"].isin(key_sources)]
    
    radar_metrics = {
        "red_red_sent_len_pct": "Satzkürzung",
        "red_red_passive_pct": "Passiv-Tilgung",
        "red_red_genitive_pct": "Genitiv-Tilgung",
        "red_red_subord_pct": "Nebensatz-Kürzung",
        "red_red_nominal_pct": "Nominalstil-Kürzung"
    }
    
    grouped = df_sub.groupby("source")[list(radar_metrics.keys())].mean().clip(lower=0, upper=100)
    categories = list(radar_metrics.values())
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = sns.color_palette("tab10", len(key_sources))
    for idx, (source, row) in enumerate(grouped.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=source, color=colors[idx])
        ax.fill(angles, values, color=colors[idx], alpha=0.1)

    plt.xticks(angles[:-1], categories, size=11, fontweight="bold")
    ax.set_rlabel_position(30)
    plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="grey", size=9)
    plt.ylim(0, 100)
    plt.title("Regel-Adhärenz Profil im Domänenvergleich (Reduktionsleistung in %)", size=14, fontweight="bold", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    out_path = os.path.join(OUTPUT_DIR, "rule_adherence_sources_radar.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    corpus_csv = "data/analysis/rule_adherence_corpus.csv"
    ladder_csv = "data/analysis/rule_adherence_ladder_sft.csv"
    
    print("Generating visualizations...")
    plot_corpus_source_side_by_side(corpus_csv)
    plot_corpus_source_percentages(corpus_csv)
    plot_source_radar(corpus_csv)
    plot_comprehensive_model_dashboard(ladder_csv)
    print("Done!")


if __name__ == "__main__":
    main()
