#!/bin/bash
set -e
#SBATCH --job-name=5_eval_token_length_exp
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/token_length/%x_%j.out
#SBATCH --error=results/logs/experiments/token_length/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi



mkdir -p results/logs/experiments/token_length results/plots/experiments/token_length results/evaluation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/evaluation

srun python scripts/evaluation/evaluate_token_length_experiment.py \
    --test_data_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --base_model_name "facebook/mbart-large-50" \
    --prompt_prefix "" \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --reward_model_path "results/models/token_length_exp/bilstm_mixup_regression_1024.pt" \
    --reward_vocab_path "data/token_length_exp/mixup_vocab_1024.json" \
    --output_summary "results/evaluation/token_length_comparison_summary.csv" \
    --output_details "results/evaluation/token_length_comparison_detailed.csv" \
    --output_metric_summary "results/evaluation/token_length_metric_comparison.csv" \
    --output_metric_details "results/evaluation/token_length_metric_details.csv"
