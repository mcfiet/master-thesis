#!/bin/bash
#SBATCH --job-name=train_sentence_regressor
#SBATCH --output=results/logs/experiments/sentencewise_regression/%x_%j.out
#SBATCH --error=results/logs/experiments/sentencewise_regression/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:mig_24gb:1

# Environment aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p results/logs/experiments/sentencewise_regression
mkdir -p results/plots/experiments/sentencewise_regression
mkdir -p results/models

python3 scripts/modeling/regression_train_sentence_mixup.py \
    --csv_path data/analysis/corpus_master.csv \
    --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --batch_size 64 \
    --embedding_dim 128 \
    --hidden_dim 128 \
    --lr 0.001 \
    --epochs 10 \
    --max_seq_len 100 \
    --min_sim 0.80 \
    --max_sim 0.98 \
    --min_sent_len 4 \
    --mixtures_per_sentence 3 \
    --model_save_path results/models/bilstm_sentence_regressor.pt \
    --vocab_save_path data/vocabs/sentence_regressor_vocab.json \
    --log_dir results/logs/experiments/sentencewise_regression \
    --plot_dir results/plots/experiments/sentencewise_regression
