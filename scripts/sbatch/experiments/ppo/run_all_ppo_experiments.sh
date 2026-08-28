#!/bin/bash
#SBATCH --job-name=run_all_ppo
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/experiments/ppo/%x_%j.out
#SBATCH --error=results/logs/experiments/ppo/%x_%j.err

set -e
# ==============================================================================
# Master Pipeline Runner: PPO Experiments & 7-Way Benchmark
# ==============================================================================
# Automates:
#   1. Decoder-Only PPO Training (Qwen 2.5)
#   2. Seq2Seq PPO Training (mBART-50)
#   3. 7-Way Master Benchmark Evaluation on Lebenshilfe test set
# ==============================================================================


echo "========================================================================"
echo "Submitting PPO Training & Evaluation Pipeline to SLURM Cluster"
echo "========================================================================"

mkdir -p results/logs/experiments/ppo/decoder_only
mkdir -p results/logs/experiments/ppo/seq2seq
mkdir -p results/plots/experiments/ppo/decoder_only
mkdir -p results/plots/experiments/ppo/seq2seq
mkdir -p results/models/decoder_only/ppo
mkdir -p results/models/ppo/seq2seq
mkdir -p results/evaluation

# 1. Submit Decoder-Only PPO Training
echo "[1/3] Submitting Decoder-Only PPO Training Job..."
JOB_DEC=$(sbatch --parsable scripts/sbatch/experiments/ppo/1_train_ppo_decoder_only.sh)
echo "  -> Decoder-Only PPO Job ID: ${JOB_DEC}"

# 2. Submit Seq2Seq PPO Training
echo "[2/3] Submitting Seq2Seq PPO Training Job..."
JOB_SEQ=$(sbatch --parsable scripts/sbatch/experiments/ppo/1_train_ppo_seq2seq.sh)
echo "  -> Seq2Seq PPO Job ID: ${JOB_SEQ}"

# 3. Submit 7-Way Benchmark Evaluation with Dependency
echo "[3/3] Submitting 7-Way Master Evaluation Job (runs after training finishes)..."
JOB_EVAL=$(sbatch --parsable --dependency=afterok:${JOB_DEC}:${JOB_SEQ} scripts/sbatch/experiments/ppo/2_evaluate_all_ppo.sh)
echo "  -> 7-Way Benchmark Evaluation Job ID: ${JOB_EVAL}"

echo "========================================================================"
echo "All PPO experiment jobs successfully scheduled in SLURM queue!"
echo "Monitor with: squeue -u \$USER"
echo "========================================================================"
