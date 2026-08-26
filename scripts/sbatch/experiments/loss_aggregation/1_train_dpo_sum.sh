#!/bin/bash
#SBATCH --job-name=train_dpo_sum
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/loss_aggregation/%x_%j.out
#SBATCH --error=results/logs/experiments/loss_aggregation/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


mkdir -p results/models/loss_aggregation_exp/dpo_sum
mkdir -p results/logs/experiments/loss_aggregation results/plots/experiments/loss_aggregation results/evaluation

echo "=== Training DPO Model with Loss Type: SUM (Classic Summed Log-Probabilities) ==="
date

TRAIN_FILE="data/corpus/dpo_pairs_mixup.jsonl"
EVAL_FILE="data/corpus/dpo_pairs_mixup_eval.jsonl"

srun python scripts/modeling/train_dpo.py \
    --model_name_or_path "results/models/sft" \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_dir "results/models/loss_aggregation_exp/dpo_sum" \
    --loss_type "sum" \
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

echo "=== DPO Training (SUM) Completed ==="
date
