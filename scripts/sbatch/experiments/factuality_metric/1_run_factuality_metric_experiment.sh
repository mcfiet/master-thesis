#!/bin/bash
set -e
#SBATCH --job-name=eval_factuality_metric
#SBATCH --partition=research
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/factuality_metric/%x_%j.out
#SBATCH --error=results/logs/experiments/factuality_metric/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/logs/experiments/factuality_metric results/plots/experiments/factuality_metric results/evaluation

echo "=== Starte Faktenkonsistenz- & Halluzinationserkennungs-Benchmark ==="
date

srun python scripts/experiments/run_factuality_metric_experiment.py

echo "=== Faktenkonsistenz-Benchmark erfolgreich abgeschlossen ==="
date
