#!/bin/bash
#SBATCH --job-name=2_eval_sft_scaling
#SBATCH --partition=research
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=results/logs/sft_scaling_eval_%j.log

set -e

echo "=== Aggregiere SFT Data Scaling Ergebnisse ==="
srun python scripts/experiments/sft_scaling/evaluate_all_sft_scaling.py \
    --results_dir results/experiments/sft_scaling \
    --output_csv results/experiments/sft_scaling/sft_scaling_summary.csv

echo "=== Aggregation abgeschlossen! ==="
