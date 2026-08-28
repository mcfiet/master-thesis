#!/bin/bash
set -e
#SBATCH --job-name=1_train_sft_decoder_only
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_48gb:1
#SBATCH --output=results/logs/experiments/decoder_only/%x_%j.out
#SBATCH --error=results/logs/experiments/decoder_only/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi



mkdir -p results/logs/experiments/decoder_only results/plots/experiments/decoder_only results/evaluation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Starting Decoder-Only SFT Training (48GB GPU) ==="
date

srun python scripts/modeling/decoder_only/train_sft_decoder_only.py \
    --corpus_path "data/analysis/corpus_master.json" \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --output_dir "results/models/decoder_only/sft" \
    --min_sim 0.70 \
    --max_sim 0.98 \
    --max_seq_length 2048 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --epochs 3 \
    --lr 1e-4 \
    --warmup_ratio 0.10 \
    --use_peft \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.10

echo "=== SFT Training Complete ==="
date
