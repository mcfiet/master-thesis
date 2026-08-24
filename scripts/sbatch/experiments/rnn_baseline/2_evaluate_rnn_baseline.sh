#!/bin/bash
#SBATCH --job-name=eval_rnn_baseline
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

mkdir -p results/evaluation results/logs

echo "=== BiLSTM vs. Vanilla RNN Baseline Evaluation auf Lebenshilfe ==="
srun python scripts/evaluation/evaluate_bilstm_vs_rnn.py \
    --test_data_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --bilstm_model_path "results/models/bilstm_mixup_regression.pt" \
    --rnn_model_path "results/models/rnn_vanilla_mixup_regression.pt" \
    --output_csv "results/evaluation/bilstm_vs_rnn_eval.csv"

echo "=== RNN Baseline Evaluation Completed ==="
date
