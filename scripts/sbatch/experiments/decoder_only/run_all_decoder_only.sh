#!/bin/bash
#SBATCH --job-name=run_all_decoder_only
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/experiments/decoder_only/%x_%j.out
#SBATCH --error=results/logs/experiments/decoder_only/%x_%j.err

set -e
# ==============================================================================
# Master Pipeline Runner for Decoder-Only SFT & DPO Experiment
# ==============================================================================
# Submits all pipeline stages sequentially via SLURM dependencies:
#   1. SFT Training (SFTTrainer + LoRA)
#   2. DPO Dataset Generation (Candidate Sampling + Reward Scoring)
#   3. DPO Training (DPOTrainer + Shared Ref Model)
#   4. Comprehensive Evaluation (NLP & Leichte Sprache Rules)
# ==============================================================================


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p results/logs/experiments/decoder_only results/plots/experiments/decoder_only results/models/decoder_only data/dpo results/evaluation

echo "=================================================================="
echo " Launching Decoder-Only SFT & DPO Pipeline"
echo "=================================================================="

# Check if sbatch is available (Cluster) or run locally
if command -v sbatch &> /dev/null; then
    echo "[1/4] Submitting SFT Training Job..."
    JOB_SFT=$(sbatch --parsable "${SCRIPT_DIR}/1_train_sft_decoder_only.sh")
    echo "  -> SFT Job ID: ${JOB_SFT}"

    echo "[2/4] Submitting DPO Dataset Generation Job (dependency: ${JOB_SFT})..."
    JOB_GEN=$(sbatch --parsable --dependency=afterok:${JOB_SFT} "${SCRIPT_DIR}/2_generate_dpo_pairs_decoder_only.sh")
    echo "  -> DPO Dataset Job ID: ${JOB_GEN}"

    echo "[3/4] Submitting DPO Training Job (dependency: ${JOB_GEN})..."
    JOB_DPO=$(sbatch --parsable --dependency=afterok:${JOB_GEN} "${SCRIPT_DIR}/3_train_dpo_decoder_only.sh")
    echo "  -> DPO Training Job ID: ${JOB_DPO}"

    echo "[4/4] Submitting Evaluation Job (dependency: ${JOB_DPO})..."
    JOB_EVAL=$(sbatch --parsable --dependency=afterok:${JOB_DPO} "${SCRIPT_DIR}/4_evaluate_decoder_only.sh")
    echo "  -> Evaluation Job ID: ${JOB_EVAL}"

    echo "=================================================================="
    echo " All jobs queued successfully!"
    echo " Monitor with: squeue -u \$USER"
    echo " Track logs in: results/logs/"
    echo "=================================================================="
else
    echo "SLURM not detected. Running pipeline stages sequentially in current shell..."
    bash "${SCRIPT_DIR}/1_train_sft_decoder_only.sh"
    bash "${SCRIPT_DIR}/2_generate_dpo_pairs_decoder_only.sh"
    bash "${SCRIPT_DIR}/3_train_dpo_decoder_only.sh"
    bash "${SCRIPT_DIR}/4_evaluate_decoder_only.sh"
fi
