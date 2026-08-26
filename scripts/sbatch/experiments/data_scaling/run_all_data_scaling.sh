#!/bin/bash
#SBATCH --job-name=run_all_data_scaling
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/experiments/data_scaling/%x_%j.out
#SBATCH --error=results/logs/experiments/data_scaling/%x_%j.err
# ==============================================================================
# Pipeline Runner: Data Scaling & Learning Curve Experiment
# Submits all dependent SLURM jobs sequentially.
# ==============================================================================

set -e

mkdir -p results/logs/experiments/data_scaling
SCRIPT_DIR="scripts/sbatch/experiments/data_scaling"

echo "=== Submitting Data Scaling Experiments ==="

# 1. Submit Grid Jobs in Parallel
JOB_MIXTURES=$(sbatch --parsable ${SCRIPT_DIR}/1_scaling_mixtures_grid.sh)
JOB_PAIRS=$(sbatch --parsable ${SCRIPT_DIR}/2_scaling_articles_grid.sh)
echo "1. Grid Jobs submitted: Mixtures Grid=$JOB_MIXTURES, Article Pairs Grid=$JOB_PAIRS"

# 2. Evaluation Job (Dependent on both training grids)
JOB_EVAL=$(sbatch --parsable --dependency=afterok:${JOB_MIXTURES}:${JOB_PAIRS} ${SCRIPT_DIR}/3_evaluate_scaling.sh)
echo "2. Evaluation Job submitted: EVAL=$JOB_EVAL (Dependent on $JOB_MIXTURES and $JOB_PAIRS)"

echo "=== All Data Scaling jobs submitted successfully! ==="
