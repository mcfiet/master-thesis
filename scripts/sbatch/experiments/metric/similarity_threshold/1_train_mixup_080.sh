#!/bin/bash
#SBATCH --job-name=train_mixup_sim_080
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

echo "=== Start MixUp Training: min_sim = 0.80 ==="
date

srun python scripts/experiments/similarity_threshold/train_similarity_mixup.py \
    --csv_path data/analysis/corpus_master.csv \
    --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --min_sim 0.80 \
    --max_sim 0.98 \
    --max_seq_len 1024 \
    --mixtures_per_pair 160 \
    --batch_size 64 \
    --embedding_dim 128 \
    --hidden_dim 128 \
    --epochs 80 \
    --patience 15 \
    --lr 0.001 \
    --output_dir results/experiments/similarity_threshold \
    --experiment_name mixup_sim_080

echo "=== MixUp Training (0.80) abgeschlossen ==="
date
