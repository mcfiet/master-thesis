#!/bin/bash
#SBATCH --job-name=enrich_glossary
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/logs/experiments/glossary/%x_%j.out
#SBATCH --error=results/logs/experiments/glossary/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi



mkdir -p results/logs/experiments/glossary results/plots/experiments/glossary results/evaluation
srun python scripts/experiments/glossary/enrich_glossary.py
