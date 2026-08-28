#!/bin/bash
set -e
#SBATCH --job-name=train_dpo_beta_005
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/dpo_beta_sweep/%x_%j.out
#SBATCH --error=results/logs/experiments/dpo_beta_sweep/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


mkdir -p results/models/dpo_beta_sweep/dpo_beta_005
mkdir -p results/logs/experiments/dpo_beta_sweep results/plots/experiments/dpo_beta_sweep results/evaluation

echo "=== Training DPO Model with Beta = 0.05 (Moderate Alignment) ==="
date

TRAIN_FILE="data/metric_weights_exp/dpo_pairs_w05_w05.jsonl"
EVAL_FILE="data/metric_weights_exp/dpo_pairs_w05_w05_eval.jsonl"

srun python scripts/modeling/train_dpo.py \
    --model_name_or_path "results/models/sft" \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_dir "results/models/dpo_beta_sweep/dpo_beta_005" \
    --loss_type "sum" \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --beta 0.05 \
    --learning_rate 5e-6 \
    --epochs 3 \
    --batch_size 2 \
    --accumulation_steps 8 \
    --patience 3 \
    --max_source_len 256 \
    --max_target_len 256 \
    --log_dir "results/logs/experiments/dpo_beta_sweep" \
    --plot_dir "results/plots/experiments/dpo_beta_sweep"

echo "=== DPO Training (Beta=0.05) Completed ==="
date
