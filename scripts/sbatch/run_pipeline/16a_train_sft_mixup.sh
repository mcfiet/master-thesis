#!/bin/bash
#SBATCH --job-name=16a_train_sft_mixup
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1

srun python scripts/modeling/train_sft.py \
    --lh_dataset_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --corpus_path "data/analysis/corpus_master.json" \
    --output_dir "results/models/sft" \
    --min_sim 0.70 \
    --max_sim 0.98 \
    --max_source_len 256 \
    --max_target_len 256 \
    --model_name "facebook/mbart-large-50" \
    --batch_size 8 \
    --accumulation_steps 2 \
    --epochs 30 \
    --lr 1e-5 \
    --warmup_ratio 0.10 \
    --patience 5 \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json"
