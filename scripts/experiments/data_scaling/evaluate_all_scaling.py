#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Aggregator: Collect and summarize all Data Scaling experiment runs
=============================================================================
Collects JSON metrics from results/experiments/data_scaling/ and produces
a consolidated CSV summary and formatted markdown report.
=============================================================================
"""

import glob
import json
import os
import pandas as pd


def main():
    results_dir = "results/experiments/data_scaling"
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' does not exist yet.")
        return

    json_files = glob.glob(os.path.join(results_dir, "*_metrics.json"))
    if not json_files:
        print(f"No metric JSON files found in {results_dir}")
        return

    records = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                records.append(data)
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    df = pd.DataFrame(records)
    
    # Sort logically
    if "experiment_group" in df.columns:
        df = df.sort_values(by=["experiment_group", "num_train_article_pairs", "mixtures_per_pair"])
    
    csv_path = os.path.join(results_dir, "scaling_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Successfully consolidated {len(df)} experiment runs into: {csv_path}")

    # Print summary table
    cols = [
        "experiment_name", "experiment_group", "num_train_article_pairs",
        "mixtures_per_pair", "total_train_samples", "test_mse", "test_mae", "test_r2", "test_binary_acc", "training_time_seconds"
    ]
    available_cols = [c for c in cols if c in df.columns]
    print("\n" + "=" * 80)
    print("DATA SCALING EXPERIMENT SUMMARY")
    print("=" * 80)
    print(df[available_cols].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
