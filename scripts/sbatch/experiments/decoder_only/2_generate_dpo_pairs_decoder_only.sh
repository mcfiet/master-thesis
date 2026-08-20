#!/bin/bash
#SBATCH --job-name=2_generate_dpo_pairs_decoder_only
#SBATCH --partition=research
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_48gb:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Starting Decoder-Only DPO Dataset Generation (48GB GPU, Jina Embeddings) ==="
date

python scripts/modeling/decoder_only/generate_dpo_dataset_decoder_only.py \
    --corpus_path "data/corpus/corpus_master_with_steps.json" \
    --sft_model_path "results/models/decoder_only/sft" \
    --base_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --reward_model_path "results/models/token_length_exp/bilstm_mixup_regression_1000.pt" \
    --reward_vocab_path "data/token_length_exp/mixup_vocab_1000.json" \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --w_style 0.7 \
    --w_sem 0.3 \
    --max_source_len 2048 \
    --max_target_len 1000 \
    --output_file "data/dpo/dpo_preference_pairs_decoder_only.jsonl" \
    --num_candidates 4 \
    --min_score_margin 0.05 \
    --temperature 0.8 \
    --top_p 0.92 \
    --batch_size 4

echo "=== DPO Dataset Generation Complete ==="
date
