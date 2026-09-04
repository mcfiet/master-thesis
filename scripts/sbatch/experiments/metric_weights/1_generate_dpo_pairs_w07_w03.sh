#!/bin/bash
#SBATCH --job-name=1_gen_dpo_w07_w03
#SBATCH --partition=research
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --array=0-3
#SBATCH --output=results/logs/experiments/metric_weights/%x_%A_%a.out
#SBATCH --error=results/logs/experiments/metric_weights/%x_%A_%a.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p data/metric_weights_exp
mkdir -p results/logs/experiments/metric_weights results/plots/experiments/metric_weights results/evaluation

NUM_SHARDS=4
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
OUTPUT_FILE="data/metric_weights_exp/dpo_pairs_w07_w03_shard_${SHARD_ID}.jsonl"

echo "=== Generating DPO Preference Pairs w07_w03 (Shard ${SHARD_ID}/${NUM_SHARDS}) ==="
date

srun python scripts/modeling/generate_dpo_dataset.py \
    --corpus_path "data/corpus/corpus_10kgnad_len512_as.json" \
    --sft_model_path "results/models/sft" \
    --base_model_name "facebook/mbart-large-50" \
    --temperature_ladder 0.6 0.7 0.8 0.85 \
    --candidates_per_step 3 \
    --max_total_candidates 12 \
    --repetition_penalty 1.35 \
    --no_repeat_ngram_size 3 \
    --max_source_len 256 \
    --max_target_len 256 \
    --reward_max_seq_len 256 \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --w_style 0.7 \
    --w_sem 0.3 \
    --min_score_margin 0.05 \
    --batch_size 16 \
    --num_shards ${NUM_SHARDS} \
    --shard_id ${SHARD_ID} \
    --output_file "${OUTPUT_FILE}" \
    --val_split_ratio 0.0 \
    --resume

echo "=== Shard ${SHARD_ID} Completed ==="
date
