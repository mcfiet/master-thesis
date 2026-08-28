#!/bin/bash
#SBATCH --job-name=15c_train_rnn_baseline
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/rnn_baseline/%x_%j.out
#SBATCH --error=results/logs/experiments/rnn_baseline/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


# 1. Standard: Vanilla RNN (Elman RNN, unidirektional)

mkdir -p results/logs/experiments/rnn_baseline results/plots/experiments/rnn_baseline results/evaluation
srun python scripts/modeling/regression_train_rnn_baseline.py \
    --csv_path data/analysis/corpus_master.csv \
    --batch_size 64 \
    --embedding_dim 128 \
    --epochs 80 \
    --hidden_dim 128 \
    --lr 0.001 \
    --max_sim 0.98 \
    --min_sim 0.8 \
    --max_seq_len 256 \
    --rnn_type rnn \
    --model_save_path results/models/rnn_vanilla_mixup_regression.pt \
    --vocab_save_path data/vocabs/mixup_vocab.json

# 2. Optional: Unidirektionales LSTM (für direkten Ablation-Vergleich zu BiLSTM)
# srun python scripts/modeling/regression_train_rnn_baseline.py \
#     --csv_path data/analysis/corpus_master.csv \
#     --batch_size 64 \
#     --embedding_dim 128 \
#     --epochs 80 \
#     --hidden_dim 128 \
#     --lr 0.001 \
#     --max_sim 0.98 \
#     --min_sim 0.8 \
#     --max_seq_len 256 \
#     --rnn_type lstm \
#     --model_save_path results/models/lstm_uni_mixup_regression.pt \
#     --vocab_save_path data/vocabs/mixup_vocab.json
