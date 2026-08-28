#!/bin/bash
set -e
#SBATCH --job-name=eval_master_benchmark
#SBATCH --partition=research
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/benchmark/%x_%j.out
#SBATCH --error=results/logs/experiments/benchmark/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p results/logs/experiments/benchmark results/plots/experiments/benchmark results/evaluation

echo "=== Starte Master 5-Wege-Benchmark Evaluation auf Lebenshilfe Testset ==="
date

srun python scripts/evaluation/evaluate_all_models_benchmark.py \
    --test_data_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --sft_mbart_path "results/models/sft" \
    --dpo_mbart_path "results/models/dpo" \
    --qwen_base_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --sft_decoder_path "results/models/decoder_only/sft" \
    --dpo_decoder_path "results/models/decoder_only/dpo" \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --output_csv "results/evaluation/benchmark_5way_decoder_vs_encoder_decoder.csv" \
    --output_summary "results/evaluation/master_benchmark_summary.csv"

echo "=== Master Benchmark Evaluation erfolgreich abgeschlossen ==="
date
