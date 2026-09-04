#!/bin/bash
#SBATCH --job-name=12_generate_synthetic_steps_corpus
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/logs/experiments/synthetic_regressor/%x_%j.out
#SBATCH --error=results/logs/experiments/synthetic_regressor/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi



mkdir -p results/logs/experiments/synthetic_regressor results/plots/experiments/synthetic_regressor results/evaluation
srun python -u scripts/experiments/synthetic_regressor/generate_synthetic_steps.py \
    --input data/corpus/corpus_master_clean.json \
    --output data/corpus/corpus_master_with_steps.json \
    --url "${FLENSGEN_API_URL:-http://193.175.180.196:8000/v1/chat/completions}" \
    --token "${FLENSGEN_API_TOKEN:-<API_TOKEN>}" \
    --model "FlensGen-GPT-OSS-120B"
