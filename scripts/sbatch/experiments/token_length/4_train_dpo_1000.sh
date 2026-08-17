#!/bin/bash
#SBATCH --job-name=4_train_dpo_1000
#SBATCH --partition=research
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1

mkdir -p results/models/token_length_exp/dpo_len1000

srun python scripts/modeling/train_dpo.py \
    --model_name_or_path "results/models/token_length_exp/sft_len1000" \
    --train_file "data/token_length_exp/dpo_pairs_len1000.jsonl" \
    --eval_file "data/token_length_exp/dpo_pairs_len1000_eval.jsonl" \
    --output_dir "results/models/token_length_exp/dpo_len1000" \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --beta 0.1 \
    --learning_rate 5e-6 \
    --epochs 3 \
    --batch_size 1 \
    --accumulation_steps 16 \
    --patience 3 \
    --max_source_len 1000 \
    --max_target_len 1000
