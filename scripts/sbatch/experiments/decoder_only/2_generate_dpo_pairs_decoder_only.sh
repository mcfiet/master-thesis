#!/bin/bash
#SBATCH --job-name=2_generate_dpo_pairs_decoder_only
#SBATCH --partition=research
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

echo "=== Starting Decoder-Only DPO Dataset Generation ==="
date

python scripts/modeling/decoder_only/generate_dpo_dataset_decoder_only.py \
    --corpus_path "data/corpus/corpus_master_with_steps.json" \
    --sft_model_path "results/models/decoder_only/sft" \
    --base_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --output_file "data/dpo/dpo_preference_pairs_decoder_only.jsonl" \
    --num_candidates 4 \
    --min_score_margin 0.10 \
    --temperature 0.8 \
    --top_p 0.92 \
    --batch_size 4

echo "=== DPO Dataset Generation Complete ==="
date
