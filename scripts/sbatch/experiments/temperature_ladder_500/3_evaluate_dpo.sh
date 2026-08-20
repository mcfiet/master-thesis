#!/bin/bash
#SBATCH --job-name=3_eval_dpo_ladder_500
#SBATCH --partition=research
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

mkdir -p results/evaluation
mkdir -p results/logs

echo "=== 4. Evaluating DPO Ladder Model vs SFT Baseline (500 Tokens) ==="
date

srun python scripts/evaluation/evaluate_dpo_ladder_model.py \
    --test_data_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --sft_model_path "results/models/token_length_exp/sft_len500" \
    --dpo_model_path "results/models/temperature_ladder_500/dpo_w05_w05" \
    --base_model_name "facebook/mbart-large-50" \
    --reward_model_path "results/models/token_length_exp/bilstm_mixup_regression_500.pt" \
    --reward_vocab_path "data/token_length_exp/mixup_vocab_500.json" \
    --sbert_model_name "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" \
    --output_summary "results/evaluation/temperature_ladder_500_summary.csv" \
    --output_details "results/evaluation/temperature_ladder_500_details.csv" \
    --max_source_len 500 \
    --max_target_len 500 \
    --batch_size 4

echo "=== Evaluation Completed ==="
date
