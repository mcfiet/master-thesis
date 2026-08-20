#!/bin/bash
#SBATCH --job-name=2_train_dpo_w05_w05
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

mkdir -p results/models/metric_weights_exp/dpo_w05_w05
mkdir -p results/logs

echo "=== Training DPO Model (w_style=0.5, w_sem=0.5) ==="
date

srun python scripts/modeling/train_dpo.py \
    --model_name_or_path "results/models/sft" \
    --train_file "data/metric_weights_exp/dpo_pairs_w05_w05.jsonl" \
    --eval_file "data/metric_weights_exp/dpo_pairs_w05_w05_eval.jsonl" \
    --output_dir "results/models/metric_weights_exp/dpo_w05_w05" \
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

echo "=== DPO Training Completed ==="
date
