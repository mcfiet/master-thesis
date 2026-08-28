#!/bin/bash
#SBATCH --job-name=run_all_token_length
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/experiments/token_length/%x_%j.out
#SBATCH --error=results/logs/experiments/token_length/%x_%j.err

set -e
# ==============================================================================
# Pipeline Runner: Token Length Experiment (256 vs 512 vs 1024 Tokens)
# Submits all dependent SLURM jobs sequentially for each length track.
# ==============================================================================


mkdir -p results/logs/experiments/token_length
SCRIPT_DIR="scripts/sbatch/experiments/token_length"

echo "=== Submitting Token Length Experiments (256, 512, 1024) ==="

# 1. Metrik Training Jobs (Parallel)
JOB_M256=$(sbatch --parsable ${SCRIPT_DIR}/1_train_metric_256.sh)
JOB_M512=$(sbatch --parsable ${SCRIPT_DIR}/1_train_metric_512.sh)
JOB_M1024=$(sbatch --parsable ${SCRIPT_DIR}/1_train_metric_1024.sh)
echo "1. Metrik Training Jobs submitted: M256=$JOB_M256, M512=$JOB_M512, M1024=$JOB_M1024"

# 2. SFT Training Jobs (Dependent on respective Metric model)
JOB_SFT256=$(sbatch --parsable --dependency=afterok:${JOB_M256} ${SCRIPT_DIR}/2_train_sft_256.sh)
JOB_SFT512=$(sbatch --parsable --dependency=afterok:${JOB_M512} ${SCRIPT_DIR}/2_train_sft_512.sh)
JOB_SFT1024=$(sbatch --parsable --dependency=afterok:${JOB_M1024} ${SCRIPT_DIR}/2_train_sft_1024.sh)
echo "2. SFT Training Jobs submitted: SFT256=$JOB_SFT256, SFT512=$JOB_SFT512, SFT1024=$JOB_SFT1024"

# 3. DPO Pair Generation Jobs (Dependent on respective SFT model)
JOB_GEN256=$(sbatch --parsable --dependency=afterok:${JOB_SFT256} ${SCRIPT_DIR}/3_generate_dpo_pairs_256.sh)
JOB_GEN512=$(sbatch --parsable --dependency=afterok:${JOB_SFT512} ${SCRIPT_DIR}/3_generate_dpo_pairs_512.sh)
JOB_GEN1024=$(sbatch --parsable --dependency=afterok:${JOB_SFT1024} ${SCRIPT_DIR}/3_generate_dpo_pairs_1024.sh)
echo "3. DPO Generation Jobs submitted: GEN256=$JOB_GEN256, GEN512=$JOB_GEN512, GEN1024=$JOB_GEN1024"

# 4. DPO Training Jobs (Dependent on respective DPO Pairs dataset)
JOB_DPO256=$(sbatch --parsable --dependency=afterok:${JOB_GEN256} ${SCRIPT_DIR}/4_train_dpo_256.sh)
JOB_DPO512=$(sbatch --parsable --dependency=afterok:${JOB_GEN512} ${SCRIPT_DIR}/4_train_dpo_512.sh)
JOB_DPO1024=$(sbatch --parsable --dependency=afterok:${JOB_GEN1024} ${SCRIPT_DIR}/4_train_dpo_1024.sh)
echo "4. DPO Training Jobs submitted: DPO256=$JOB_DPO256, DPO512=$JOB_DPO512, DPO1024=$JOB_DPO1024"

# 5. Final Evaluation Job (Runs after all DPO models finish)
JOB_EVAL=$(sbatch --parsable --dependency=afterok:${JOB_DPO256}:${JOB_DPO512}:${JOB_DPO1024} ${SCRIPT_DIR}/5_run_full_evaluation.sh)
echo "5. Evaluation Job submitted: EVAL=$JOB_EVAL"

echo "=== All jobs submitted successfully with dependency chain! ==="
