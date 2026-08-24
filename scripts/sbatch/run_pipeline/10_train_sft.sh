#!/bin/bash
#SBATCH --job-name=10_train_sft
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

mkdir -p results/models/sft
mkdir -p results/logs

echo "=== Start SFT Training (mBART-50 auf corpus_master) ==="
date

srun python scripts/modeling/train_sft.py \
    --corpus_path "data/analysis/corpus_master.json" \
    --output_dir "results/models/sft" \
    --min_sim 0.70 \
    --max_sim 0.98 \
    --max_source_len 500 \
    --max_target_len 500 \
    --model_name "facebook/mbart-large-50" \
    --batch_size 8 \
    --accumulation_steps 2 \
    --epochs 30 \
    --lr 1e-4 \
    --warmup_ratio 0.10 \
    --patience 10 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json"

echo "=== SFT Training abgeschlossen ==="
date
