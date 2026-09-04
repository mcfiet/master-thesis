#!/bin/bash
#SBATCH --job-name=run_06_dpo_pipeline
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/run_pipeline/%x_%j.out
#SBATCH --error=results/logs/run_pipeline/%x_%j.err

set -e
# =============================================================================
# Themenbereich 6: DPO-Pipeline (Präferenz-Generierung & DPO-Training)
# =============================================================================
# Startet Schritt 11 (DPO Paar-Generierung mit 4 Shards), Schritt 11b (Merge)
# und Schritt 12 (LoRA DPO Training) mit Slurm Job-Abhängigkeit (--dependency=afterok).
# =============================================================================

mkdir -p results/logs/run_pipeline
mkdir -p data/corpus
mkdir -p results/models/dpo

echo "========================================================================"
echo "Starte Themenbereich 6: DPO-Pipeline (Sharded Array + Auto-Merge)..."
echo "Gestartet am: $(date)"
echo "========================================================================"

JOB11=$(sbatch --parsable scripts/sbatch/run_pipeline/11_generate_dpo_dataset.sh)
echo "Schritt 11 eingereicht (DPO Paare generieren, 4 Shards): Job ID $JOB11"

JOB11B=$(sbatch --parsable --dependency=afterok:$JOB11 scripts/sbatch/run_pipeline/11b_merge_dpo_dataset.sh)
echo "Schritt 11b eingereicht (DPO Shards zusammenführen): Job ID $JOB11B"

JOB12=$(sbatch --parsable --dependency=afterok:$JOB11B scripts/sbatch/run_pipeline/12_train_dpo.sh)
echo "Schritt 12 eingereicht (DPO Training): Job ID $JOB12"

echo "========================================================================"
echo "DPO-Pipeline erfolgreich eingereicht!"
echo "Status prüfen mit: squeue -u \$USER"
echo "Logs überwachen in: results/logs/run_pipeline/"
echo "========================================================================"
