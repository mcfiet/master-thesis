#!/bin/bash
set -e
#SBATCH --job-name=eval_dpo_beta_sweep
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/dpo_beta_sweep/%x_%j.out
#SBATCH --error=results/logs/experiments/dpo_beta_sweep/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


mkdir -p results/evaluation
mkdir -p results/plots/experiments/dpo_beta_sweep
mkdir -p results/logs/experiments/dpo_beta_sweep

echo "=== Running Comprehensive DPO Beta Sweep Evaluation ==="
date

srun python scripts/evaluation/evaluate_dpo_beta_experiment.py \
    --test_data_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --base_model_name "facebook/mbart-large-50" \
    --sft_model_path "results/models/sft" \
    --dpo_beta_001_path "results/models/dpo_beta_sweep/dpo_beta_001" \
    --dpo_beta_005_path "results/models/dpo_beta_sweep/dpo_beta_005" \
    --dpo_beta_010_path "results/models/dpo_beta_sweep/dpo_beta_010" \
    --dpo_beta_020_path "results/models/dpo_beta_sweep/dpo_beta_020" \
    --dpo_beta_050_path "results/models/dpo_beta_sweep/dpo_beta_050" \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --output_summary "results/evaluation/dpo_beta_comparison_summary.csv" \
    --output_details "results/evaluation/dpo_beta_comparison_details.csv" \
    --output_plot "results/plots/experiments/dpo_beta_sweep/dpo_beta_pareto_tradeoff.png"

echo "=== DPO Beta Sweep Evaluation Completed ==="
date
