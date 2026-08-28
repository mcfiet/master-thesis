#!/bin/bash
set -e
#SBATCH --job-name=eval_loss_aggregation
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/experiments/loss_aggregation/%x_%j.out
#SBATCH --error=results/logs/experiments/loss_aggregation/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


mkdir -p results/evaluation
mkdir -p results/plots
mkdir -p results/logs/experiments/loss_aggregation results/plots/experiments/loss_aggregation results/evaluation

echo "=== Running Full Evaluation for DPO Loss Aggregation Experiment (Sum vs. Mean) ==="
date

srun python scripts/evaluation/evaluate_loss_aggregation_experiment.py \
    --test_file "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --base_model_name "facebook/mbart-large-50" \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --output_dir "results/evaluation" \
    --plot_dir "results/plots"

echo "=== Evaluation Completed ==="
date
