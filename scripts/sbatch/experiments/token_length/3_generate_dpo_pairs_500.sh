#!/bin/bash
#SBATCH --job-name=3_generate_dpo_pairs_500
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1

mkdir -p data/token_length_exp

srun python scripts/modeling/generate_dpo_dataset.py \
    --corpus_path "data/analysis/corpus_master.json" \
    --min_sim 0.70 \
    --max_sim 0.98 \
    --sft_model_path "results/models/token_length_exp/sft_len500" \
    --prompt_prefix "" \
    --num_candidates 5 \
    --temperature 0.8 \
    --max_source_len 500 \
    --max_target_len 500 \
    --reward_max_seq_len 500 \
    --reward_model_path "results/models/token_length_exp/bilstm_mixup_regression_500.pt" \
    --reward_vocab_path "data/token_length_exp/mixup_vocab_500.json" \
    --w_style 0.5 \
    --w_sem 0.5 \
    --min_score_margin 0.05 \
    --output_file "data/token_length_exp/dpo_pairs_len500.jsonl" \
    --val_split_ratio 0.15
