#!/bin/bash
#SBATCH --job-name=3_eval_data_scaling
#SBATCH --partition=research
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=results/logs/data_scaling_eval_%j.log

set -e

echo "=== Aggregating Data Scaling Experiment Results ==="
srun python scripts/experiments/data_scaling/evaluate_all_scaling.py
echo "=== Evaluation & Aggregation Finished ==="
