#!/bin/bash
#SBATCH --job-name=18b_train_dpo_synthetic
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/synthetic_regressor/%x_%j.out
#SBATCH --error=results/logs/experiments/synthetic_regressor/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi



mkdir -p results/logs/experiments/synthetic_regressor results/plots/experiments/synthetic_regressor results/evaluation results/models/experiments/synthetic_regressor/dpo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python scripts/modeling/train_dpo.py \
    --model_name_or_path "results/models/experiments/synthetic_regressor/sft" \
    --train_file "data/dpo/dpo_preference_pairs_synthetic.jsonl" \
    --eval_file "data/dpo/dpo_preference_pairs_synthetic_eval.jsonl" \
    --output_dir "results/models/experiments/synthetic_regressor/dpo" \
    --log_dir "results/logs/experiments/synthetic_regressor" \
    --plot_dir "results/plots/experiments/synthetic_regressor" \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --beta 0.1 \
    --learning_rate 5e-6 \
    --epochs 3 \
    --batch_size 1 \
    --accumulation_steps 16 \
    --patience 3 \
    --max_source_len 1024 \
    --max_target_len 1024
