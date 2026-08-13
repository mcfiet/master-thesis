#!/bin/bash
#SBATCH --job-name=3_regression_train_mixup
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1

srun python scripts/modeling/3_regression_train_mixup.py \
        --csv_path data/analysis/corpus_master.csv \
        --batch_size 64 \
        --embedding_dim 128 \
        --epochs 40 \
        --hidden_dim 128 \
        --lr 0.001 \
        --max_sim 0.98 \
        --min_sim 0.7 \
        --max_seq_len 256
