#!/bin/bash
#SBATCH --job-name=1b_merge_dpo_w07_w03
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=results/logs/experiments/metric_weights/%x_%j.out
#SBATCH --error=results/logs/experiments/metric_weights/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p data/metric_weights_exp results/logs/experiments/metric_weights

echo "=== Merging DPO Shards (w07_w03) ==="
date

python scripts/modeling/merge_dpo_shards.py \
    --input_pattern "data/metric_weights_exp/dpo_pairs_w07_w03_shard_*.jsonl" \
    --output_file "data/metric_weights_exp/dpo_pairs_w07_w03.jsonl" \
    --val_split_ratio 0.15 \
    --cleanup_shards

echo "=== Merge Completed ==="
date
