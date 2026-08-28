#!/bin/bash
#SBATCH --job-name=1_train_metric_512
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/token_length/%x_%j.out
#SBATCH --error=results/logs/experiments/token_length/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi



mkdir -p results/logs/experiments/token_length results/plots/experiments/token_length results/evaluation
mkdir -p results/models/token_length_exp data/token_length_exp

srun python scripts/modeling/regression_train_mixup.py \
    --csv_path data/analysis/corpus_master.csv \
    --batch_size 64 \
    --embedding_dim 128 \
    --epochs 80 \
    --hidden_dim 128 \
    --lr 0.001 \
    --max_sim 0.98 \
    --min_sim 0.8 \
    --max_seq_len 512 \
    --model_save_path results/models/token_length_exp/bilstm_mixup_regression_512.pt \
    --vocab_save_path data/token_length_exp/mixup_vocab_512.json
