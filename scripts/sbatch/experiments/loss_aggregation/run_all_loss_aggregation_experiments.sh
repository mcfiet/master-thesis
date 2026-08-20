#!/bin/bash
# Master Runner for Loss Aggregation Experiment (Sum vs. Mean DPO)
# Submits training jobs concurrently and queues evaluation upon completion.

echo "========================================================================"
echo "Starte Loss-Aggregations-Experiment: Sum vs. Mean Log-Probabilities"
echo "========================================================================"

mkdir -p results/logs
mkdir -p results/models/loss_aggregation_exp/dpo_sum
mkdir -p results/models/loss_aggregation_exp/dpo_mean
mkdir -p results/evaluation
mkdir -p results/plots

# 1. Starte Generierung der sauberen DPO-Paare (reines Self-Play ohne Human Ground Truth)
JOB_GEN=$(sbatch --parsable scripts/sbatch/experiments/metric_weights/1_generate_dpo_pairs_w05_w05.sh)
echo "[SLURM] Gestartet: DPO-Paare Generierung -> Job-ID: $JOB_GEN"

# 2. Starte Training Jobs (warten auf fertige DPO-Paare)
JOB_SUM=$(sbatch --parsable --dependency=afterok:${JOB_GEN} scripts/sbatch/experiments/loss_aggregation/1_train_dpo_sum.sh)
echo "[SLURM] Eingeplant: DPO Sum Training -> Job-ID: $JOB_SUM (wartet auf $JOB_GEN)"

JOB_MEAN=$(sbatch --parsable --dependency=afterok:${JOB_GEN} scripts/sbatch/experiments/loss_aggregation/1_train_dpo_mean.sh)
echo "[SLURM] Eingeplant: DPO Mean Training -> Job-ID: $JOB_MEAN (wartet auf $JOB_GEN)"

# 3. Starte Evaluation nach erfolgreichem Training beider Modelle
JOB_EVAL=$(sbatch --parsable --dependency=afterok:${JOB_SUM}:${JOB_MEAN} scripts/sbatch/experiments/loss_aggregation/2_run_full_evaluation.sh)
echo "[SLURM] Eingeplant: Full Evaluation -> Job-ID: $JOB_EVAL (wartet auf $JOB_SUM und $JOB_MEAN)"

echo "========================================================================"
echo "Alle Jobs erfolgreich verkettet an Slurm übergeben!"
echo "Status prüfen mit: squeue -u \$USER"
echo "========================================================================"
