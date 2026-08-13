#!/bin/bash
#SBATCH --job-name=4_regression_train_synthetic
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1

srun python scripts/modeling/4_regression_train_synthetic.py \
        --corpus_with_steps_path data/corpus/corpus_master_with_steps.json \
        --lh_with_steps_path data/lebenshilfe/lebenshilfe_dataset_with_steps.json \
        --model_save_path results/models/bilstm_regressor_synthetic.pt \
        --vocab_save_path data/vocabs/synthetic_vocab.json \
        --epochs 40 \
        --max_seq_len 256 \
        --min_sim 0.70 \
        --max_sim 0.98
