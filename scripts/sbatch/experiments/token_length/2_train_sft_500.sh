#!/bin/bash
#SBATCH --job-name=2_train_sft_500
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1

mkdir -p results/models/token_length_exp/sft_len500

srun python scripts/modeling/train_sft.py \
    --lh_dataset_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --corpus_path "data/analysis/corpus_master.json" \
    --output_dir "results/models/token_length_exp/sft_len500" \
    --min_sim 0.70 \
    --max_sim 0.98 \
    --max_source_len 500 \
    --max_target_len 500 \
    --model_name "facebook/mbart-large-50" \
    --batch_size 4 \
    --accumulation_steps 4 \
    --epochs 30 \
    --lr 1e-4 \
    --warmup_ratio 0.10 \
    --patience 10 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --reward_model_path "results/models/token_length_exp/bilstm_mixup_regression_500.pt" \
    --reward_vocab_path "data/token_length_exp/mixup_vocab_500.json" \
    --reward_max_seq_len 500
