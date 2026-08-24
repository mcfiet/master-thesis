#!/bin/bash
# =============================================================================
# Themenbereich 5: SFT-Training (Supervised Fine-Tuning)
# =============================================================================
# Startet Schritt 10 (mBART-50 SFT Training auf corpus_master.json).
# =============================================================================

set -e

mkdir -p results/logs/run_pipeline
mkdir -p results/models/sft

echo "========================================================================"
echo "Starte Themenbereich 5: SFT-Training..."
echo "Gestartet am: $(date)"
echo "========================================================================"

JOB10=$(sbatch --parsable scripts/sbatch/run_pipeline/10_train_sft.sh)
echo "Schritt 10 eingereicht (SFT Training): Job ID $JOB10"

echo "========================================================================"
echo "SFT-Training erfolgreich eingereicht!"
echo "Status prüfen mit: squeue -u \$USER"
echo "Logs überwachen in: results/logs/run_pipeline/"
echo "========================================================================"
