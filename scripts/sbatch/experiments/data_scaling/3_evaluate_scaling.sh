#!/bin/bash
#SBATCH --job-name=3_eval_data_scaling
#SBATCH --partition=research
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=results/logs/experiments/data_scaling/%x_%j.out
#SBATCH --error=results/logs/experiments/data_scaling/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


mkdir -p results/logs/experiments/data_scaling results/plots/experiments/data_scaling results/evaluation

echo "=== Aggregating Data Scaling Experiment Results ==="
srun python scripts/experiments/data_scaling/evaluate_all_scaling.py

echo "=== Running Zero-Shot Evaluation on Lebenshilfe Gold Dataset ==="
srun python scripts/experiments/data_scaling/evaluate_scaling_on_lebenshilfe.py

echo "=== Evaluation & Aggregation Finished ==="
