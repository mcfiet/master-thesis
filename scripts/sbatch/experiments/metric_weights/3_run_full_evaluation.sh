#!/bin/bash
#SBATCH --job-name=3_eval_metric_weights
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/metric_weights/%x_%j.out
#SBATCH --error=results/logs/experiments/metric_weights/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


mkdir -p results/evaluation
mkdir -p results/plots
mkdir -p results/logs/experiments/metric_weights results/plots/experiments/metric_weights results/evaluation

echo "=== Running Metric Weights Evaluation (0.5/0.5 vs 0.7/0.3 vs 1.0/0.0) ==="
date

srun python scripts/evaluation/evaluate_metric_weights_experiment.py \
    --test_data_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --base_model_name "facebook/mbart-large-50" \
    --sft_model_path "results/models/sft" \
    --dpo_w05_w05_path "results/models/metric_weights_exp/dpo_w05_w05" \
    --dpo_w07_w03_path "results/models/metric_weights_exp/dpo_w07_w03" \
    --dpo_w10_w00_path "results/models/metric_weights_exp/dpo_w10_w00" \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --output_summary "results/evaluation/metric_weights_comparison_summary.csv" \
    --output_details "results/evaluation/metric_weights_comparison_details.csv" \
    --output_plot "results/plots/metric_weights_tradeoff_curve.png"

echo "=== Evaluation Completed ==="
date
