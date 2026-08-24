#!/bin/bash
#SBATCH --job-name=4_train_dpo_500
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/token_length/%x_%j.out
#SBATCH --error=results/logs/experiments/token_length/%x_%j.err


mkdir -p results/logs/experiments/token_length results/plots/experiments/token_length results/evaluation
mkdir -p results/models/token_length_exp/dpo_len500

srun python scripts/modeling/train_dpo.py \
    --model_name_or_path "results/models/token_length_exp/sft_len500" \
    --train_file "data/token_length_exp/dpo_pairs_len500.jsonl" \
    --eval_file "data/token_length_exp/dpo_pairs_len500_eval.jsonl" \
    --output_dir "results/models/token_length_exp/dpo_len500" \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --beta 0.1 \
    --learning_rate 5e-6 \
    --epochs 3 \
    --batch_size 2 \
    --accumulation_steps 8 \
    --patience 3 \
    --max_source_len 500 \
    --max_target_len 500
