#!/bin/bash
set -e
#SBATCH --job-name=eval_context_length_ablation
#SBATCH --partition=research
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/experiments/context_length_ablation/%x_%j.out
#SBATCH --error=results/logs/experiments/context_length_ablation/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "$HOME/master-thesis/.venv" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/logs/experiments/context_length_ablation results/plots/experiments/context_length_ablation results/evaluation data/analysis

echo "=== Starte Jina Kontextlängen-Ablationsstudie (128 vs 256 vs 512 vs 1024 vs 8192) ==="
date

srun python scripts/evaluation/evaluate_jina_context_ablation.py \
    --input_path "data/analysis/corpus_master.json" \
    --output_csv "data/analysis/jina_context_ablation.csv" \
    --summary_csv "results/evaluation/jina_context_ablation_summary.csv" \
    --model_name "jinaai/jina-embeddings-v2-base-de"

echo "=== Kontextlängen-Ablationsstudie erfolgreich abgeschlossen ==="
date
