#!/bin/bash
#SBATCH --job-name=train_dpo_mean
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

mkdir -p results/models/loss_aggregation_exp/dpo_mean
mkdir -p results/logs

echo "=== Training DPO Model with Loss Type: MEAN (Length-Normalized / Per-Token Log-Probabilities) ==="
date

# Check for training dataset (fallback to metric_weights_exp if default not present)
TRAIN_FILE="data/metric_weights_exp/dpo_pairs_w05_w05.jsonl"
EVAL_FILE="data/metric_weights_exp/dpo_pairs_w05_w05_eval.jsonl"
if [ ! -f "$TRAIN_FILE" ]; then
    TRAIN_FILE="data/dpo_preference_pairs.jsonl"
    EVAL_FILE="data/dpo_preference_pairs_eval.jsonl"
fi

srun python scripts/modeling/train_dpo.py \
    --model_name_or_path "results/models/sft" \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_dir "results/models/loss_aggregation_exp/dpo_mean" \
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
    --max_source_len 256 \
    --max_target_len 256

echo "=== DPO Training (MEAN) Completed ==="
date
