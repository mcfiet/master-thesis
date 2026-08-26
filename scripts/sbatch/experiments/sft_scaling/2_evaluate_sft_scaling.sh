#!/bin/bash
#SBATCH --job-name=2_eval_sft_scaling
#SBATCH --partition=research
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=results/logs/experiments/sft_scaling/%x_%j.out
#SBATCH --error=results/logs/experiments/sft_scaling/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


mkdir -p results/logs/experiments/sft_scaling results/plots/experiments/sft_scaling results/evaluation
set -e

echo "=== Aggregiere SFT Data Scaling Ergebnisse ==="
srun python scripts/experiments/sft_scaling/evaluate_all_sft_scaling.py \
    --results_dir results/experiments/sft_scaling \
    --output_csv results/experiments/sft_scaling/sft_scaling_summary.csv

echo "=== Aggregation abgeschlossen! ==="
