#!/bin/bash
#SBATCH --job-name=mt5_experiment
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/logs/experiments/mt5_exp/mt5_pipeline_%j.out
#SBATCH --error=results/logs/experiments/mt5_exp/mt5_pipeline_%j.err
# =============================================================================
# SLURM Job Script: Google mT5-base End-to-End Pipeline (SFT -> DPO -> Benchmark)
# =============================================================================

set -e

mkdir -p results/logs/experiments/mt5_exp
mkdir -p results/plots/experiments/mt5_exp

echo "========================================================================"
echo " SLURM JOB: mT5-base Pipeline (SFT & DPO)"
echo " Job ID:    $SLURM_JOB_ID"
echo " Node:      $SLURMD_NODENAME"
echo " Startzeit: $(date)"
echo "========================================================================"

# Ausführung des modularen Bash-Runners
bash scripts/experiments/run_mt5_experiment.sh --all

echo "========================================================================"
echo " SLURM JOB abgeschlossen am $(date)"
echo "========================================================================"
