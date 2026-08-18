#!/bin/bash
#SBATCH --job-name=4_train_dpo_500_jina
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_48gb:1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/models/token_length_jina_exp/dpo_len500_jina

srun python scripts/modeling/train_dpo.py \
    --model_name_or_path "results/models/token_length_exp/sft_len500" \
    --train_file "data/token_length_jina_exp/dpo_pairs_len500_jina.jsonl" \
    --eval_file "data/token_length_jina_exp/dpo_pairs_len500_jina_eval.jsonl" \
    --output_dir "results/models/token_length_jina_exp/dpo_len500_jina" \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --beta 0.1 \
    --learning_rate 5e-6 \
    --epochs 3 \
    --batch_size 2 \
    --accumulation_steps 8 \
    --patience 3 \
    --max_source_len 500 \
    --max_target_len 500
