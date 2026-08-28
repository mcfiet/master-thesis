#!/bin/bash
set -e
#SBATCH --job-name=build_expert_eval_pool
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/benchmark/%x_%j.out
#SBATCH --error=results/logs/experiments/benchmark/%x_%j.err

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p results/logs/experiments/benchmark data/expert_eval results/expert_eval

echo "=== Erstelle Experten-Evaluationsdatensatz (10 Nicht-Lebenshilfe Domaenen) ==="
date

srun python scripts/evaluation/build_expert_evaluation_set.py \
    --testset_csv "data/evaluation_sets/benchmark_translation_testset.csv" \
    --sft_mbart_path "results/models/sft" \
    --dpo_mbart_path "results/models/dpo" \
    --qwen_base_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --output_dir "data/expert_eval" \
    --num_articles 10 \
    --seed 42

echo "=== Experten-Evaluationsdatensatz erfolgreich erstellt ==="
date
