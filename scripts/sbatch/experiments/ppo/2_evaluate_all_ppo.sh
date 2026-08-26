#!/bin/bash
#SBATCH --job-name=2_evaluate_all_ppo
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_48gb:1
#SBATCH --output=results/logs/experiments/ppo/%x_%j.out
#SBATCH --error=results/logs/experiments/ppo/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


mkdir -p results/logs/experiments/ppo results/plots/experiments/ppo results/evaluation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Starting 7-Way Benchmark Evaluation (Few-Shot, SFT, DPO, PPO) ==="
date

srun python scripts/evaluation/evaluate_ppo_experiment.py \
    --test_data_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --qwen_base_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --sft_decoder_path "results/models/decoder_only/sft" \
    --dpo_decoder_path "results/models/decoder_only/dpo" \
    --ppo_decoder_path "results/models/decoder_only/ppo" \
    --mbart_base_model "facebook/mbart-large-50" \
    --sft_mbart_path "results/models/sft" \
    --dpo_mbart_path "results/models/dpo" \
    --ppo_mbart_path "results/models/ppo/seq2seq" \
    --output_csv "results/evaluation/benchmark_ppo_vs_dpo_vs_sft_7way.csv" \
    --output_summary "results/evaluation/master_benchmark_summary_7way.csv" \
    --plot_dir "results/plots/experiments/ppo" \
    --log_dir "results/logs/experiments/ppo"

echo "=== 7-Way Benchmark Evaluation Complete ==="
date
