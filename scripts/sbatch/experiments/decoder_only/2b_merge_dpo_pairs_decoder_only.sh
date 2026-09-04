#!/bin/bash
#SBATCH --job-name=2b_merge_dpo_dec
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=results/logs/experiments/decoder_only/%x_%j.out
#SBATCH --error=results/logs/experiments/decoder_only/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p data/dpo results/logs/experiments/decoder_only

echo "=== Merging Decoder-Only DPO Shards ==="
date

python scripts/modeling/merge_dpo_shards.py \
    --input_pattern "data/dpo/dpo_preference_pairs_decoder_only_shard_*.jsonl" \
    --output_file "data/dpo/dpo_preference_pairs_decoder_only.jsonl" \
    --val_split_ratio 0.15 \
    --cleanup_shards

echo "=== Merge Completed ==="
date
