#!/bin/bash
#SBATCH --job-name=eval_sim_ablation
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=results/logs/experiments/similarity_threshold/%x_%j.out
#SBATCH --error=results/logs/experiments/similarity_threshold/%x_%j.err

set -e

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p results/logs/experiments/similarity_threshold results/experiments/similarity_threshold/plots
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

echo "=== Starte Konsolidierung und Gesamtevaluation der Similarity-Experimente ==="
date

srun python scripts/experiments/similarity_threshold/evaluate_all_similarity_thresholds.py \
    --results_dir "results/experiments/similarity_threshold" \
    --plots_dir "results/experiments/similarity_threshold/plots" \
    --corpus_path "data/analysis/corpus_master.csv"

echo "=== Evaluation abgeschlossen ==="
date
