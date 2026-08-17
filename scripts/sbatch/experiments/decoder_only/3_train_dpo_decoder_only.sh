#!/bin/bash
#SBATCH --job-name=3_train_dpo_decoder_only
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

echo "=== Starting Decoder-Only DPO Training with TRL ==="
date

python scripts/modeling/decoder_only/train_dpo_decoder_only.py \
    --dpo_train_file "data/dpo/dpo_preference_pairs_decoder_only.jsonl" \
    --sft_model_path "results/models/decoder_only/sft" \
    --base_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_dir "results/models/decoder_only/dpo" \
    --beta 0.1 \
    --lr 5e-6 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --epochs 3 \
    --warmup_ratio 0.10 \
    --max_length 1024 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05

echo "=== DPO Training Complete ==="
date
