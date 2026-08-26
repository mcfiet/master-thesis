#!/bin/bash
#SBATCH --job-name=run_all_beta_sweep
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/experiments/dpo_beta_sweep/%x_%j.out
#SBATCH --error=results/logs/experiments/dpo_beta_sweep/%x_%j.err
# ==============================================================================
# Pipeline Runner: DPO Beta Sweep Experiment (Beta = 0.01, 0.05, 0.10, 0.20, 0.50)
# Submits parallel training jobs for all Beta configurations and schedules
# the evaluation job to run after all training jobs complete.
# ==============================================================================

set -e

mkdir -p results/logs/experiments/dpo_beta_sweep
SCRIPT_DIR="scripts/sbatch/experiments/dpo_beta_sweep"

echo "=== Submitting DPO Beta Sweep Experiment Jobs ==="

# 1. DPO Training Jobs (Parallel across Beta configurations)
JOB_BETA_001=$(sbatch --parsable ${SCRIPT_DIR}/1_train_dpo_beta_001.sh)
JOB_BETA_005=$(sbatch --parsable ${SCRIPT_DIR}/1_train_dpo_beta_005.sh)
JOB_BETA_010=$(sbatch --parsable ${SCRIPT_DIR}/1_train_dpo_beta_010.sh)
JOB_BETA_020=$(sbatch --parsable ${SCRIPT_DIR}/1_train_dpo_beta_020.sh)
JOB_BETA_050=$(sbatch --parsable ${SCRIPT_DIR}/1_train_dpo_beta_050.sh)

echo "1. DPO Training Jobs submitted:"
echo "   - Beta 0.01: Job ID $JOB_BETA_001"
echo "   - Beta 0.05: Job ID $JOB_BETA_005"
echo "   - Beta 0.10: Job ID $JOB_BETA_010"
echo "   - Beta 0.20: Job ID $JOB_BETA_020"
echo "   - Beta 0.50: Job ID $JOB_BETA_050"

# 2. Final Evaluation Job (Runs after all 5 DPO models finish training)
JOB_EVAL=$(sbatch --parsable --dependency=afterok:${JOB_BETA_001}:${JOB_BETA_005}:${JOB_BETA_010}:${JOB_BETA_020}:${JOB_BETA_050} ${SCRIPT_DIR}/2_run_full_evaluation.sh)
echo "2. Evaluation Job scheduled with dependency: Job ID $JOB_EVAL"

echo "=== All jobs submitted successfully with dependency chain! ==="
