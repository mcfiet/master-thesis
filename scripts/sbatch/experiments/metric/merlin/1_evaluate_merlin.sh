#!/bin/bash
#SBATCH --job-name=eval_merlin
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/merlin/%x_%j.out
#SBATCH --error=results/logs/experiments/merlin/%x_%j.err

set -e

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/logs/experiments/merlin results/plots/experiments/merlin results/evaluation data/analysis/merlin

echo "=== Starte MERLIN CEFR Benchmark Evaluation (Regressoren & Klassifikatoren) ==="
date

srun python scripts/evaluation/evaluate_merlin.py \
    --benchmark_json "data/analysis/merlin/merlin_de.json" \
    --benchmark_csv "data/analysis/merlin/merlin_texts.csv" \
    --output_csv "results/evaluation/merlin_all_models_eval.csv" \
    --summary_json "results/evaluation/merlin_summary.json" \
    --plot_dir "results/plots/experiments/merlin"

echo "=== MERLIN Evaluation erfolgreich abgeschlossen ==="
date
