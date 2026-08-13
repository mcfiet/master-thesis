#!/bin/bash
#SBATCH --job-name=5_train_sft
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:full:1

srun python scripts/modeling/5_train_sft.py \
    --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --corpus_path data/analysis/corpus_master.json \
    --min_sim 0.70 --max_sim 0.98 --max_source_len 256 --max_target_len 256 \
    --model_name facebook/mbart-large-50 \
    --batch_size 8 --epochs 40 --lr 1e-5 --warmup_ratio 0.10 \
    --patience 5 --seed 42 --val_split 0.15 --output_dir results/models/sft \
    --reward_model_path results/models/bilstm_synthetic_regression.pt \
    --reward_vocab_path data/vocabs/synthetic_vocab.json
