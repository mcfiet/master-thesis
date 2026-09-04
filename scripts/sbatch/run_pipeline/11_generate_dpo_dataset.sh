#!/bin/bash
#SBATCH --job-name=11_generate_dpo_dataset
#SBATCH --partition=research
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --array=0-3
#SBATCH --output=results/logs/run_pipeline/%x_%A_%a.out
#SBATCH --error=results/logs/run_pipeline/%x_%A_%a.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p data/corpus
mkdir -p results/logs/run_pipeline results/plots/run_pipeline results/evaluation
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

NUM_SHARDS=4
SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
OUTPUT_FILE="data/corpus/dpo_pairs_mixup_shard_${SHARD_ID}.jsonl"

echo "=== Start DPO-Präferenzdatengenerierung (Shard ${SHARD_ID}/${NUM_SHARDS}) ==="
date

srun python scripts/modeling/generate_dpo_dataset.py \
    --corpus_path "data/corpus/corpus_10kgnad_len512_as.json" \
    --sft_model_path "results/models/sft" \
    --base_model_name "facebook/mbart-large-50" \
    --max_source_len 1024 \
    --max_target_len 1024 \
    --reward_max_seq_len 1024 \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --temperature_ladder 0.6 0.7 0.8 0.85 \
    --candidates_per_step 3 \
    --max_total_candidates 12 \
    --repetition_penalty 1.2 \
    --no_repeat_ngram_size 3 \
    --min_score_margin 0.05 \
    --w_style 0.5 \
    --w_sem 0.5 \
    --batch_size 8 \
    --num_shards ${NUM_SHARDS} \
    --shard_id ${SHARD_ID} \
    --output_file "${OUTPUT_FILE}" \
    --val_split_ratio 0.0 \
    --resume

echo "=== Shard ${SHARD_ID} abgeschlossen ==="
date
