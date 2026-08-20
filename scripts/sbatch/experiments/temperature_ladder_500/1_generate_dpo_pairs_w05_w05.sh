#!/bin/bash
#SBATCH --job-name=1_gen_dpo_ladder_500
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

mkdir -p data/temperature_ladder_500
mkdir -p results/logs

echo "=== 2. Generating DPO Preference Pairs via Temperature Ladder (500 Tokens) ==="
date

srun python scripts/modeling/generate_dpo_dataset_ladder.py \
    --corpus_path "data/temperature_ladder_500/corpus_10kgnad_len500_as.json" \
    --sft_model_path "results/models/token_length_exp/sft_len500" \
    --base_model_name "facebook/mbart-large-50" \
    --max_source_len 500 \
    --max_target_len 500 \
    --reward_max_seq_len 500 \
    --reward_model_path "results/models/token_length_exp/bilstm_mixup_regression_500.pt" \
    --reward_vocab_path "data/token_length_exp/mixup_vocab_500.json" \
    --sbert_model_name "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" \
    --temperature_ladder 0.7 0.8 0.9 1.0 \
    --candidates_per_step 3 \
    --max_total_candidates 12 \
    --min_score_margin 0.05 \
    --w_style 0.5 \
    --w_sem 0.5 \
    --batch_size 4 \
    --output_file "data/temperature_ladder_500/dpo_pairs_w05_w05.jsonl" \
    --val_split_ratio 0.15

echo "=== DPO Generation Completed ==="
date
