#!/bin/bash
#SBATCH --job-name=4_evaluate_decoder_only
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_48gb:1
#SBATCH --output=results/logs/experiments/decoder_only/%x_%j.out
#SBATCH --error=results/logs/experiments/decoder_only/%x_%j.err


mkdir -p results/logs/experiments/decoder_only results/plots/experiments/decoder_only results/evaluation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Starting Decoder-Only Evaluation (48GB GPU, Jina Embeddings) ==="
date

mkdir -p results/evaluation

# 1. Evaluate SFT Model
echo "Evaluating SFT Baseline..."
python scripts/modeling/decoder_only/evaluate_decoder_only.py \
    --test_data_path "data/corpus/corpus_master_with_steps.json" \
    --model_path "results/models/decoder_only/sft" \
    --base_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --sbert_model "jinaai/jina-embeddings-v2-base-de" \
    --max_target_len 1500 \
    --output_file "results/evaluation/eval_sft_decoder_only.csv" \
    --max_samples 100

# 2. Evaluate DPO Model
echo "Evaluating DPO Optimized Model..."
python scripts/modeling/decoder_only/evaluate_decoder_only.py \
    --test_data_path "data/corpus/corpus_master_with_steps.json" \
    --model_path "results/models/decoder_only/dpo" \
    --base_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --sbert_model "jinaai/jina-embeddings-v2-base-de" \
    --max_target_len 1500 \
    --output_file "results/evaluation/eval_dpo_decoder_only.csv" \
    --max_samples 100

echo "=== Evaluation Complete ==="
date
