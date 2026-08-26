#!/bin/bash
#SBATCH --job-name=run_03_corpus_building
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/run_pipeline/%x_%j.out
#SBATCH --error=results/logs/run_pipeline/%x_%j.err
# =============================================================================
# Themenbereich 3: Korpus-Erstellung (Master-Korpus & 10kGNAD)
# =============================================================================
# Startet Schritt 05 (build_corpus_master) und Schritt 06 (10kGNAD DPO-Korpus).
# =============================================================================

set -e

mkdir -p results/logs/run_pipeline
mkdir -p data/corpus
mkdir -p data/analysis

echo "========================================================================"
echo "Starte Themenbereich 3: Korpus-Erstellung..."
echo "Gestartet am: $(date)"
echo "========================================================================"

JOB5=$(sbatch --parsable scripts/sbatch/run_pipeline/05_build_corpus_master.sh)
echo "Schritt 05 eingereicht (Corpus Master erstellen): Job ID $JOB5"

JOB6=$(sbatch --parsable scripts/sbatch/run_pipeline/06_prepare_10kgnad_dpo_corpus.sh)
echo "Schritt 06 eingereicht (10kGNAD DPO-Korpus vorbereiten): Job ID $JOB6"

echo "========================================================================"
echo "Korpus-Erstellung erfolgreich eingereicht!"
echo "Status prüfen mit: squeue -u \$USER"
echo "Logs überwachen in: results/logs/run_pipeline/"
echo "========================================================================"
