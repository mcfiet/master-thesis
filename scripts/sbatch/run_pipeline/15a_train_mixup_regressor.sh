#!/bin/bash
#SBATCH --job-name=15a_train_mixup_regressor
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1

srun python scripts/modeling/regression_train_mixup.py \
    --csv_path data/analysis/corpus_master.csv \
    --batch_size 64 \
    --embedding_dim 128 \
    --epochs 40 \
    --hidden_dim 128 \
    --lr 0.001 \
    --max_sim 0.98 \
    --min_sim 0.8 \
    --max_seq_len 256 \
    --model_save_path results/models/bilstm_mixup_regression.pt \
    --vocab_save_path data/vocabs/mixup_vocab.json
