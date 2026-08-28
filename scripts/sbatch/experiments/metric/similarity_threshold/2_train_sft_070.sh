#!/bin/bash
#SBATCH --job-name=train_sft_sim_070
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/similarity_threshold/%x_%j.out
#SBATCH --error=results/logs/experiments/similarity_threshold/%x_%j.err

set -e

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p results/logs/experiments/similarity_threshold results/experiments/similarity_threshold
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

echo "=== Start SFT Training (mBART-50): min_sim = 0.70 ==="
date

srun python scripts/experiments/similarity_threshold/train_similarity_sft.py \
    --corpus_path "data/analysis/corpus_master.csv" \
    --lh_dataset_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --min_sim 0.70 \
    --max_sim 0.98 \
    --max_source_len 1024 \
    --max_target_len 1024 \
    --base_model_name "facebook/mbart-large-50" \
    --batch_size 2 \
    --accumulation_steps 8 \
    --epochs 30 \
    --lr 1e-4 \
    --warmup_ratio 0.10 \
    --patience 10 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --output_dir "results/experiments/similarity_threshold" \
    --experiment_name "sft_sim_070"

echo "=== SFT Training (0.70) abgeschlossen ==="
date
