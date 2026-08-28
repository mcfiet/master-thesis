#!/bin/bash
#SBATCH --job-name=run_all_metric_weights
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/experiments/metric_weights/%x_%j.out
#SBATCH --error=results/logs/experiments/metric_weights/%x_%j.err

set -e
# ==============================================================================
# Pipeline Runner: Metric Weighting Experiment (0.5/0.5 vs. 0.7/0.3 vs. 1.0/0.0)
# Submits all dependent SLURM jobs sequentially for all weighting configurations.
# ==============================================================================


mkdir -p results/logs/experiments/metric_weights
SCRIPT_DIR="scripts/sbatch/experiments/metric_weights"

echo "=== Submitting Metric Weighting Experiment Jobs ==="

# 1. DPO Preference Pair Generation Jobs (Parallel across weight schemes)
JOB_GEN_05=$(sbatch --parsable ${SCRIPT_DIR}/1_generate_dpo_pairs_w05_w05.sh)
JOB_GEN_07=$(sbatch --parsable ${SCRIPT_DIR}/1_generate_dpo_pairs_w07_w03.sh)
JOB_GEN_10=$(sbatch --parsable ${SCRIPT_DIR}/1_generate_dpo_pairs_w10_w00.sh)
echo "1. DPO Pair Generation Jobs submitted: w05_w05=$JOB_GEN_05, w07_w03=$JOB_GEN_07, w10_w00=$JOB_GEN_10"

# 2. DPO Training Jobs (Dependent on respective dataset generation)
JOB_DPO_05=$(sbatch --parsable --dependency=afterok:${JOB_GEN_05} ${SCRIPT_DIR}/2_train_dpo_w05_w05.sh)
JOB_DPO_07=$(sbatch --parsable --dependency=afterok:${JOB_GEN_07} ${SCRIPT_DIR}/2_train_dpo_w07_w03.sh)
JOB_DPO_10=$(sbatch --parsable --dependency=afterok:${JOB_GEN_10} ${SCRIPT_DIR}/2_train_dpo_w10_w00.sh)
echo "2. DPO Training Jobs submitted: DPO_05=$JOB_DPO_05, DPO_07=$JOB_DPO_07, DPO_10=$JOB_DPO_10"

# 3. Final Comparative Evaluation Job (Runs after all 3 DPO models finish training)
JOB_EVAL=$(sbatch --parsable --dependency=afterok:${JOB_DPO_05}:${JOB_DPO_07}:${JOB_DPO_10} ${SCRIPT_DIR}/3_run_full_evaluation.sh)
echo "3. Evaluation Job submitted: EVAL=$JOB_EVAL"

echo "=== All jobs submitted successfully with dependency chain! ==="
