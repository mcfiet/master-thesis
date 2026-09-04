#!/bin/bash
#SBATCH --job-name=11b_merge_dpo_dataset
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=results/logs/run_pipeline/%x_%j.out
#SBATCH --error=results/logs/run_pipeline/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p data/corpus
mkdir -p results/logs/run_pipeline results/plots/run_pipeline results/evaluation

echo "=== Zusammenführen der DPO-Shards (Hauptpipeline) ==="
date

python scripts/modeling/merge_dpo_shards.py \
    --input_pattern "data/corpus/dpo_pairs_mixup_shard_*.jsonl" \
    --output_file "data/corpus/dpo_pairs_mixup.jsonl" \
    --val_split_ratio 0.15 \
    --cleanup_shards

echo "=== Zusammenführen abgeschlossen ==="
date
