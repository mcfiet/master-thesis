#!/bin/bash
#SBATCH --job-name=run_all_sim_experiments
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/experiments/similarity_threshold/%x_%j.out
#SBATCH --error=results/logs/experiments/similarity_threshold/%x_%j.err

set -e
# ==============================================================================
# Pipeline Runner: Similarity Threshold Ablation (0.60 vs. 0.70 vs. 0.80)
# Submits all MixUp and SFT jobs in parallel, followed by the evaluation job.
# ==============================================================================

mkdir -p results/logs/experiments/similarity_threshold
SCRIPT_DIR="scripts/sbatch/experiments/similarity_threshold"

echo "=== Submitting Similarity Threshold Experiments (MixUp + SFT) ==="

# 1. Submit MixUp Regressor Grid
JOB_MIX_060=$(sbatch --parsable ${SCRIPT_DIR}/1_train_mixup_060.sh)
JOB_MIX_070=$(sbatch --parsable ${SCRIPT_DIR}/1_train_mixup_070.sh)
JOB_MIX_080=$(sbatch --parsable ${SCRIPT_DIR}/1_train_mixup_080.sh)
echo "1. MixUp Jobs submitted: 0.60=$JOB_MIX_060, 0.70=$JOB_MIX_070, 0.80=$JOB_MIX_080"

# 2. Submit SFT mBART Grid
JOB_SFT_060=$(sbatch --parsable ${SCRIPT_DIR}/2_train_sft_060.sh)
JOB_SFT_070=$(sbatch --parsable ${SCRIPT_DIR}/2_train_sft_070.sh)
JOB_SFT_080=$(sbatch --parsable ${SCRIPT_DIR}/2_train_sft_080.sh)
echo "2. SFT Jobs submitted: 0.60=$JOB_SFT_060, 0.70=$JOB_SFT_070, 0.80=$JOB_SFT_080"

# 3. Submit Consolidation & Evaluation Job (Dependent on all 6 training runs)
DEP_LIST="${JOB_MIX_060}:${JOB_MIX_070}:${JOB_MIX_080}:${JOB_SFT_060}:${JOB_SFT_070}:${JOB_SFT_080}"
JOB_EVAL=$(sbatch --parsable --dependency=afterok:${DEP_LIST} ${SCRIPT_DIR}/3_evaluate_all.sh)
echo "3. Evaluation Job submitted: EVAL=$JOB_EVAL (Dependent on all training runs)"

echo "=== All Similarity Threshold jobs successfully submitted! ==="
