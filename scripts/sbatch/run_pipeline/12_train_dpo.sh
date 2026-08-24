#!/bin/bash
#SBATCH --job-name=12_train_dpo
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

mkdir -p results/models/dpo
mkdir -p results/logs

echo "=== Start DPO Training (LoRA auf mBART-50 SFT) ==="
date

srun python scripts/modeling/train_dpo.py \
    --model_name_or_path "results/models/sft" \
    --train_file "data/corpus/dpo_pairs_mixup.jsonl" \
    --eval_file "data/corpus/dpo_pairs_mixup_eval.jsonl" \
    --output_dir "results/models/dpo" \
    --loss_type "mean" \
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

echo "=== DPO Training abgeschlossen ==="
date
