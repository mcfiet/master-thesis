#!/usr/bin/env python3
"""
Aggregations- und Auswertungsskript für das SFT Data Scaling Experiment.
Sammelt alle *_metrics.json Dateien ein und generiert eine zusammenfassende CSV-Tabelle.
"""

import os
import glob
import json
import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Aggregate SFT Data Scaling results.")
    parser.add_argument("--results_dir", type=str, default="results/experiments/sft_scaling")
    parser.add_argument("--output_csv", type=str, default="results/experiments/sft_scaling/sft_scaling_summary.csv")
    args = parser.parse_args()

    json_files = glob.glob(os.path.join(args.results_dir, "*_metrics.json"))
    if not json_files:
        print(f"[WARNUNG] Keine Metrik-Dateien (*_metrics.json) in '{args.results_dir}' gefunden.")
        return

    records = []
    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            records.append(data)
        except Exception as e:
            print(f"[FEHLER] Konnte {fpath} nicht lesen: {e}")

    df = pd.DataFrame(records)
    if "train_fraction" in df.columns:
        df = df.sort_values(by="train_fraction").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[ERFOLG] Aggregierte Zusammenfassung ({len(df)} Läufe) gespeichert unter: {args.output_csv}\n")

    cols_to_print = [
        "experiment_name", "train_fraction", "num_train_pairs",
        "r_style_mean", "r_sem_as_mean", "sim_ref_mean",
        "composite_reward_mean", "bleu_mean", "rouge_l_mean",
        "avg_gen_tokens", "truncation_rate_pct", "training_time_seconds"
    ]
    avail_cols = [c for c in cols_to_print if c in df.columns]
    
    print("=== SFT DATA SCALING ERGEBNIS-TABELLE ===")
    styled_df = df[avail_cols].copy()
    for col in styled_df.columns:
        if col not in ["experiment_name", "num_train_pairs"]:
            styled_df[col] = styled_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else x)
            
    print(styled_df.to_string(index=False))


if __name__ == "__main__":
    main()
