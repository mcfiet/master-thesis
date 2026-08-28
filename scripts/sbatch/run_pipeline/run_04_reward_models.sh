#!/bin/bash
#SBATCH --job-name=run_04_reward_models
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/run_pipeline/%x_%j.out
#SBATCH --error=results/logs/run_pipeline/%x_%j.err

set -e
# =============================================================================
# Themenbereich 4: Reward-Modelle & Klassifikatoren
# =============================================================================
# Startet Schritt 07 (Satz-Klassifikator), Schritt 08 (Artikel-Klassifikator)
# und Schritt 09 (MixUp-Regressor) parallel auf GPUs.
# =============================================================================


mkdir -p results/logs/run_pipeline
mkdir -p data/vocabs
mkdir -p results/models

echo "========================================================================"
echo "Starte Themenbereich 4: Reward-Modelle & Klassifikatoren..."
echo "Gestartet am: $(date)"
echo "========================================================================"

JOB7=$(sbatch --parsable scripts/sbatch/run_pipeline/07_train_sentence_classifier.sh)
echo "Schritt 07 eingereicht (Satz-Klassifikator): Job ID $JOB7"

JOB8=$(sbatch --parsable scripts/sbatch/run_pipeline/08_train_article_classifier.sh)
echo "Schritt 08 eingereicht (Artikel-Klassifikator): Job ID $JOB8"

JOB9=$(sbatch --parsable scripts/sbatch/run_pipeline/09_train_mixup_regressor.sh)
echo "Schritt 09 eingereicht (MixUp-Regressor): Job ID $JOB9"

echo "========================================================================"
echo "Reward-Modelle erfolgreich parallel eingereicht!"
echo "Status prüfen mit: squeue -u \$USER"
echo "Logs überwachen in: results/logs/run_pipeline/"
echo "========================================================================"
