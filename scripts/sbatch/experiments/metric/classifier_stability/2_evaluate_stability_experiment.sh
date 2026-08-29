#!/bin/bash
#SBATCH --job-name=eval_clf_stability
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=results/logs/experiments/classifier_stability/%x_%j.out
#SBATCH --error=results/logs/experiments/classifier_stability/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p results/logs/experiments/classifier_stability results/plots/experiments/classifier_stability results/evaluation/classifier_stability
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

echo "========================================================================"
echo "STARTING CLASSIFIER STABILITY EVALUATION"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "========================================================================"

python scripts/experiments/classifier_stability/evaluate_classifier_stability.py \
    --eval_dir results/evaluation/classifier_stability \
    --plot_dir results/plots/experiments/classifier_stability

echo "========================================================================"
echo "EVALUATION COMPLETE."
echo "========================================================================"
