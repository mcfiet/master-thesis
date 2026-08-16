#!/bin/bash
#SBATCH --job-name=19_train_dpo
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:full:1

srun python scripts/modeling/train_dpo.py \
    --model_name_or_path "results/models/sft" \
    --train_file "data/dpo_preference_pairs.jsonl" \
    --eval_file "data/dpo_preference_pairs_eval.jsonl" \
    --output_dir "results/models/dpo_trained_model" \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --beta 0.1 \
    --learning_rate 5e-6 \
    --epochs 3 \
    --batch_size 2 \
    --accumulation_steps 8 \
    --patience 3 \
    --max_source_len 256 \
    --max_target_len 256
