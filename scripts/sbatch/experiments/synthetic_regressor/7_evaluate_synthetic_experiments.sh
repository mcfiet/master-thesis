#!/bin/bash
#SBATCH --job-name=eval_synthetic_experiments
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/synthetic_regressor/%x_%j.out
#SBATCH --error=results/logs/experiments/synthetic_regressor/%x_%j.err

mkdir -p results/logs/experiments/synthetic_regressor results/plots/experiments/synthetic_regressor results/evaluation

echo "=== 1. Evaluierung MixUp vs. Synthetic Regressor (Unbiased) ==="
srun python scripts/evaluation/evaluate_mixup_vs_synthetic.py \
    --mixup_model_path "results/models/bilstm_mixup_regression.pt" \
    --synthetic_model_path "results/models/bilstm_synthetic_regression.pt" \
    --steps_dataset_path "data/lebenshilfe/lebenshilfe_dataset_with_steps.json" \
    --output_csv "results/evaluation/mixup_vs_synthetic_unbiased_eval.csv"

echo "=== 2. Evaluierung Score-Verteilungen & KDE (Lebenshilfe) ==="
srun python scripts/evaluation/evaluate_mixup_synthetic_kde.py \
    --test_data_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --mixup_model_path "results/models/bilstm_mixup_regression.pt" \
    --synthetic_model_path "results/models/bilstm_synthetic_regression.pt" \
    --output_csv "results/evaluation/mixup_synthetic_kde_eval.csv"

echo "=== Synthetic Evaluations Completed ==="
date
