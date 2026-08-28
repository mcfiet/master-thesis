#!/bin/bash
#SBATCH --job-name=3_train_dpo_decoder_only
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/decoder_only/%x_%j.out
#SBATCH --error=results/logs/experiments/decoder_only/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi



mkdir -p results/logs/experiments/decoder_only results/plots/experiments/decoder_only results/evaluation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Starting Decoder-Only DPO Training with TRL (48GB GPU) ==="
date

srun python scripts/modeling/decoder_only/train_dpo_decoder_only.py \
    --dpo_train_file "data/dpo/dpo_preference_pairs_decoder_only.jsonl" \
    --sft_model_path "results/models/decoder_only/sft" \
    --base_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --output_dir "results/models/decoder_only/dpo" \
    --beta 0.1 \
    --lr 2e-6 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --epochs 3 \
    --warmup_ratio 0.10 \
    --max_length 2048 \
    --use_peft \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.10

echo "=== DPO Training Complete ==="
date
