#!/bin/bash
set -e
#SBATCH --job-name=eval_textcomplexityde
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/textcomplexityde/%x_%j.out
#SBATCH --error=results/logs/experiments/textcomplexityde/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/logs/experiments/textcomplexityde results/plots/experiments/textcomplexityde results/evaluation

echo "=== Starte TextComplexityDE Benchmark Evaluation ==="
date

srun python scripts/evaluation/evaluate_textcomplexityde.py \
    --model_path "results/models/bilstm_mixup_regression.pt" \
    --vocab_path "data/vocabs/mixup_vocab.json" \
    --benchmark_csv "data/analysis/textcomplexityde/ratings.csv" \
    --output_csv "results/evaluation/textcomplexityde_eval.csv"

echo "=== TextComplexityDE Evaluation erfolgreich abgeschlossen ==="
date
