#!/bin/bash
#SBATCH --job-name=eval_length_bias
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_48gb:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/evaluation results/logs

echo "=== Starte Length-Bias & Shortcut Evaluation ==="
date

srun python scripts/evaluation/check_length_bias.py \
    --dataset_path "data/lebenshilfe/lebenshilfe_dataset_no_paragraphs.json" \
    --model_path "results/models/lstm_article_sim_0.80_to_0.98.pt" \
    --vocab_source_csv "data/analysis/information_loss_analysis_cleaned.csv" \
    --output_csv "results/evaluation/length_bias_results.csv"

echo "=== Length-Bias Evaluation erfolgreich abgeschlossen ==="
date
