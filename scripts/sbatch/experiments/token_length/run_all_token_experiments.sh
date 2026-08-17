#!/bin/bash
# ==============================================================================
# Pipeline Runner: Token Length Experiment (256 vs 500 vs 1000 Tokens)
# Submits all dependent SLURM jobs sequentially for each length track.
# ==============================================================================

set -e

SCRIPT_DIR="scripts/sbatch/experiments/token_length"

echo "=== Submitting Token Length Experiments ==="

# 1. Metrik Training Jobs (Parallel)
JOB_M256=$(sbatch --parsable ${SCRIPT_DIR}/1_train_metric_256.sh)
JOB_M500=$(sbatch --parsable ${SCRIPT_DIR}/1_train_metric_500.sh)
JOB_M1000=$(sbatch --parsable ${SCRIPT_DIR}/1_train_metric_1000.sh)
echo "1. Metrik Training Jobs submitted: M256=$JOB_M256, M500=$JOB_M500, M1000=$JOB_M1000"

# 2. SFT Training Jobs (Dependent on respective Metric model)
JOB_SFT256=$(sbatch --parsable --dependency=afterok:${JOB_M256} ${SCRIPT_DIR}/2_train_sft_256.sh)
JOB_SFT500=$(sbatch --parsable --dependency=afterok:${JOB_M500} ${SCRIPT_DIR}/2_train_sft_500.sh)
JOB_SFT1000=$(sbatch --parsable --dependency=afterok:${JOB_M1000} ${SCRIPT_DIR}/2_train_sft_1000.sh)
echo "2. SFT Training Jobs submitted: SFT256=$JOB_SFT256, SFT500=$JOB_SFT500, SFT1000=$JOB_SFT1000"

# 3. DPO Pair Generation Jobs (Dependent on respective SFT model)
JOB_GEN256=$(sbatch --parsable --dependency=afterok:${JOB_SFT256} ${SCRIPT_DIR}/3_generate_dpo_pairs_256.sh)
JOB_GEN500=$(sbatch --parsable --dependency=afterok:${JOB_SFT500} ${SCRIPT_DIR}/3_generate_dpo_pairs_500.sh)
JOB_GEN1000=$(sbatch --parsable --dependency=afterok:${JOB_SFT1000} ${SCRIPT_DIR}/3_generate_dpo_pairs_1000.sh)
echo "3. DPO Generation Jobs submitted: GEN256=$JOB_GEN256, GEN500=$JOB_GEN500, GEN1000=$JOB_GEN1000"

# 4. DPO Training Jobs (Dependent on respective DPO Pairs dataset)
JOB_DPO256=$(sbatch --parsable --dependency=afterok:${JOB_GEN256} ${SCRIPT_DIR}/4_train_dpo_256.sh)
JOB_DPO500=$(sbatch --parsable --dependency=afterok:${JOB_GEN500} ${SCRIPT_DIR}/4_train_dpo_500.sh)
JOB_DPO1000=$(sbatch --parsable --dependency=afterok:${JOB_GEN1000} ${SCRIPT_DIR}/4_train_dpo_1000.sh)
echo "4. DPO Training Jobs submitted: DPO256=$JOB_DPO256, DPO500=$JOB_DPO500, DPO1000=$JOB_DPO1000"

# 5. Final Evaluation Job (Runs after all DPO models finish)
JOB_EVAL=$(sbatch --parsable --dependency=afterok:${JOB_DPO256}:${JOB_DPO500}:${JOB_DPO1000} ${SCRIPT_DIR}/5_run_full_evaluation.sh)
echo "5. Evaluation Job submitted: EVAL=$JOB_EVAL"

echo "=== All jobs submitted successfully with dependency chain! ==="
