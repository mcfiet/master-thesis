#!/bin/bash
set -e
#SBATCH --job-name=3_generate_dpo_pairs_256
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/token_length/%x_%j.out
#SBATCH --error=results/logs/experiments/token_length/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi



mkdir -p results/logs/experiments/token_length results/plots/experiments/token_length results/evaluation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p data/token_length_exp

srun python scripts/modeling/generate_dpo_dataset.py \
    --corpus_path "data/corpus/corpus_10kgnad_len512_as.json" \
    --sft_model_path "results/models/token_length_exp/sft_len256" \
    --base_model_name "facebook/mbart-large-50" \
    --temperature_ladder 0.6 0.7 0.8 0.85 \
    --candidates_per_step 3 \
    --max_total_candidates 12 \
    --repetition_penalty 1.35 \
    --no_repeat_ngram_size 3 \
    --max_source_len 256 \
    --max_target_len 256 \
    --reward_max_seq_len 256 \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --reward_model_path "results/models/token_length_exp/bilstm_mixup_regression_256.pt" \
    --reward_vocab_path "data/token_length_exp/mixup_vocab_256.json" \
    --w_style 0.5 \
    --w_sem 0.5 \
    --min_score_margin 0.05 \
    --batch_size 16 \
    --output_file "data/token_length_exp/dpo_pairs_len256.jsonl" \
    --val_split_ratio 0.15 \
    --resume
