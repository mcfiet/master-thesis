#!/bin/bash
# =============================================================================
# Themenbereich 7: Pipeline-Evaluierung
# =============================================================================
# Startet Schritt 13 (Finale Benchmark-Evaluierung auf dem ungesehenen
# Lebenshilfe-Datensatz).
# =============================================================================

set -e

mkdir -p results/logs/run_pipeline
mkdir -p results/evaluation

echo "========================================================================"
echo "Starte Themenbereich 7: Pipeline-Evaluierung..."
echo "Gestartet am: $(date)"
echo "========================================================================"

JOB13=$(sbatch --parsable scripts/sbatch/run_pipeline/13_evaluate_pipeline.sh)
echo "Schritt 13 eingereicht (Pipeline-Evaluierung): Job ID $JOB13"

echo "========================================================================"
echo "Pipeline-Evaluierung erfolgreich eingereicht!"
echo "Status prüfen mit: squeue -u \$USER"
echo "Logs überwachen in: results/logs/run_pipeline/"
echo "========================================================================"
