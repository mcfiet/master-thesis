#!/bin/bash
# =============================================================================
# Run All: Temperature Ladder 500-Token Experiment Pipeline
# =============================================================================
# Chains:
#   Step 0: Prepare 10kGNAD 500-Token Corpus
#   Step 1: Generate DPO Preference Pairs via Temperature Ladder
#   Step 2: Train DPO Model (500 Tokens, Loss: MEAN)
#   Step 3: Evaluate DPO Model vs SFT Baseline on Lebenshilfe Benchmark
# =============================================================================

set -e

mkdir -p results/logs
mkdir -p data/temperature_ladder_500
mkdir -p results/models/temperature_ladder_500
mkdir -p results/evaluation

echo "========================================================================"
echo "Submitting Temperature Ladder 500-Token Pipeline..."
echo "========================================================================"

# Step 0: Prepare Corpus
JOB0=$(sbatch --parsable scripts/sbatch/experiments/temperature_ladder_500/0_prepare_corpus.sh)
echo "Submitted Step 0 (Prepare Corpus): Job ID $JOB0"

# Step 1: Generate DPO Pairs (depends on Step 0)
JOB1=$(sbatch --parsable --dependency=afterok:$JOB0 scripts/sbatch/experiments/temperature_ladder_500/1_generate_dpo_pairs_w10_w00.sh)
echo "Submitted Step 1 (Generate DPO Pairs): Job ID $JOB1"

# Step 2: Train DPO Model (depends on Step 1)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/sbatch/experiments/temperature_ladder_500/2_train_dpo_w10_w00.sh)
echo "Submitted Step 2 (Train DPO Model): Job ID $JOB2"

# Step 3: Evaluate DPO Model (depends on Step 2)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 scripts/sbatch/experiments/temperature_ladder_500/3_evaluate_dpo.sh)
echo "Submitted Step 3 (Evaluate DPO Model): Job ID $JOB3"

echo "========================================================================"
echo "Full Pipeline Submitted Successfully!"
echo "Track progress with: squeue -u \$USER"
echo "View logs in: results/logs/"
echo "========================================================================"
