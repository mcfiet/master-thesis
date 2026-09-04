#!/bin/bash
#SBATCH --job-name=3b_merge_dpo_512
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=results/logs/experiments/token_length/%x_%j.out
#SBATCH --error=results/logs/experiments/token_length/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p results/logs/experiments/token_length data/token_length_exp

echo "=== Merging DPO Shards (Token Length 512) ==="
date

python scripts/modeling/merge_dpo_shards.py \
    --input_pattern "data/token_length_exp/dpo_pairs_len512_shard_*.jsonl" \
    --output_file "data/token_length_exp/dpo_pairs_len512.jsonl" \
    --val_split_ratio 0.15 \
    --cleanup_shards

echo "=== Merge Completed ==="
date
