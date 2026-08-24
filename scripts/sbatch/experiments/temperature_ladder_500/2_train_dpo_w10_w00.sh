#!/bin/bash
#SBATCH --job-name=2_train_dpo_ladder_500
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

mkdir -p results/models/temperature_ladder_500/dpo_w10_w00
mkdir -p results/logs

echo "=== 3. Training DPO Model (500 Tokens, Loss: MEAN, w_style=1.0, w_sem=0.0) ==="
date

srun python scripts/modeling/train_dpo.py \
    --model_name_or_path "results/models/token_length_exp/sft_len500" \
    --train_file "data/temperature_ladder_500/dpo_pairs_w10_w00.jsonl" \
    --eval_file "data/temperature_ladder_500/dpo_pairs_w10_w00_eval.jsonl" \
    --output_dir "results/models/temperature_ladder_500/dpo_w10_w00" \
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

echo "=== DPO Training Completed ==="
date
