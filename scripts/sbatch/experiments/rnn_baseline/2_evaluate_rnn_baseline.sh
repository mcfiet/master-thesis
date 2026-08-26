#!/bin/bash
#SBATCH --job-name=eval_rnn_baseline
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/rnn_baseline/%x_%j.out
#SBATCH --error=results/logs/experiments/rnn_baseline/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


mkdir -p results/logs/experiments/rnn_baseline results/plots/experiments/rnn_baseline results/evaluation

echo "=== BiLSTM vs. Vanilla RNN Baseline Evaluation auf Lebenshilfe ==="
srun python scripts/evaluation/evaluate_bilstm_vs_rnn.py \
    --test_data_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --bilstm_model_path "results/models/bilstm_mixup_regression.pt" \
    --rnn_model_path "results/models/rnn_vanilla_mixup_regression.pt" \
    --output_csv "results/evaluation/bilstm_vs_rnn_eval.csv" \
    --output_predictions_csv "results/evaluation/bilstm_vs_rnn_predictions.csv"

echo "=== RNN Baseline Evaluation Completed ==="
date
