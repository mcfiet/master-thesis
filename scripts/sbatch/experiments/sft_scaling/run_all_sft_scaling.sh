#!/bin/bash
#SBATCH --job-name=run_all_sft_scaling
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/experiments/sft_scaling/%x_%j.out
#SBATCH --error=results/logs/experiments/sft_scaling/%x_%j.err

set -e
# ==============================================================================
# Pipeline Runner: SFT Data Scaling & Learning Curve Experiment
# Startet das Grid-Training und anschließend automatisch die Evaluation
# ==============================================================================


mkdir -p results/logs/experiments/sft_scaling
SCRIPT_DIR="scripts/sbatch/experiments/sft_scaling"

echo "=== Starte SFT Data Scaling Pipeline ==="

# 1. Grid Training Job
JOB_TRAIN=$(sbatch --parsable ${SCRIPT_DIR}/1_train_sft_scaling_grid.sh)
echo "1. Grid Training Job übermittelt: JOB_ID=$JOB_TRAIN"

# 2. Evaluation Job (Abhängig vom erfolgreichen Abschluss des Grid Trainings)
JOB_EVAL=$(sbatch --parsable --dependency=afterok:${JOB_TRAIN} ${SCRIPT_DIR}/2_evaluate_sft_scaling.sh)
echo "2. Evaluations-Job übermittelt: JOB_ID=$JOB_EVAL (Abhängig von $JOB_TRAIN)"

echo "=== Alle SFT Scaling Jobs erfolgreich eingereiht! ==="
