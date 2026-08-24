#!/bin/bash
# =============================================================================
# Themenbereich 2: Lebenshilfe-Datensatz Vorbereitung & Bereinigung
# =============================================================================
# Startet Schritt 03 (Dokumente einlesen) und Schritt 04 (Bereinigung)
# mit Slurm Job-Abhängigkeit (--dependency=afterok).
# =============================================================================

set -e

mkdir -p results/logs/run_pipeline
mkdir -p data/lebenshilfe

echo "========================================================================"
echo "Starte Themenbereich 2: Lebenshilfe Vorbereitung & Bereinigung..."
echo "Gestartet am: $(date)"
echo "========================================================================"

JOB3=$(sbatch --parsable scripts/sbatch/run_pipeline/03_create_lebenshilfe_dataset.sh)
echo "Schritt 03 eingereicht (Lebenshilfe einlesen): Job ID $JOB3"

JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 scripts/sbatch/run_pipeline/04_clean_lebenshilfe.sh)
echo "Schritt 04 eingereicht (Lebenshilfe bereinigen): Job ID $JOB4"

echo "========================================================================"
echo "Lebenshilfe-Aufbereitung erfolgreich eingereicht!"
echo "Status prüfen mit: squeue -u \$USER"
echo "Logs überwachen in: results/logs/run_pipeline/"
echo "========================================================================"
