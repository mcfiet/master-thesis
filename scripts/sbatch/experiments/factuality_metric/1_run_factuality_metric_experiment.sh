#!/bin/bash
#SBATCH --job-name=eval_factuality_metric
#SBATCH --partition=research
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_48gb:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/evaluation results/logs

echo "=== Starte Faktenkonsistenz- & Halluzinationserkennungs-Benchmark ==="
date

srun python scripts/experiments/run_factuality_metric_experiment.py

echo "=== Faktenkonsistenz-Benchmark erfolgreich abgeschlossen ==="
date
