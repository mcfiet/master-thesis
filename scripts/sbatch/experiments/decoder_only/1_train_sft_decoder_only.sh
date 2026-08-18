#!/bin/bash
#SBATCH --job-name=1_train_sft_decoder_only
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_48gb:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Starting Decoder-Only SFT Training (48GB GPU) ==="
date

python scripts/modeling/decoder_only/train_sft_decoder_only.py \
    --corpus_path "data/corpus/corpus_master_with_steps.json" \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_dir "results/models/decoder_only/sft" \
    --min_sim 0.70 \
    --max_sim 0.98 \
    --max_seq_length 1024 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --epochs 5 \
    --lr 2e-4 \
    --warmup_ratio 0.05 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05

echo "=== SFT Training Complete ==="
date
