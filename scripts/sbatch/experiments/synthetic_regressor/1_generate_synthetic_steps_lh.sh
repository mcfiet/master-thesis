#!/bin/bash
#SBATCH --job-name=11_generate_synthetic_steps_lh
#SBATCH --partition=research
#SBATCH --time=06:00:00
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
    --input data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --output data/lebenshilfe/lebenshilfe_dataset_with_steps.json \
    --url http://193.175.180.196:8000/v1/chat/completions \
    --token RrI6y403jAlUm8v \
    --model "FlensGen-GPT-OSS-120B"
