#!/bin/bash
#SBATCH --job-name=6_train_dpo_synthetic
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:full:1

srun python scripts/modeling/6_train_dpo.py \
    --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --corpus_path data/analysis/corpus_master.json \
    --output_dir results/models/dpo_synthetic \
    --sft_model_dir results/models/sft \
    --reward_model_path results/models/bilstm_synthetic_regression.pt \
    --reward_vocab_path data/vocabs/synthetic_vocab.json \
    --min_sim 0.70 --max_sim 0.98 --w_style 0.5 --w_sem 0.5 \
    --max_source_len 256 --max_target_len 256 --model_name facebook/mbart-large-50 \
    --batch_size 16 \
    --accumulation_steps 2 \
    --epochs 2 \
    --lr 1e-6 \
    --beta 0.3
