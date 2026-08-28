#!/bin/bash
# ==============================================================================
# Master-Skript: MixUp Modell-Varianten Experiment Pipeline
# ==============================================================================
# Trainiert alle 4 MixUp-Varianten und startet anschließend die Evaluation.
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR/../../../.."

echo "=== Starte vollständige MixUp-Varianten Pipeline ==="

# 1. Training der 4 Varianten
JOB_STATIC=$(sbatch --parsable scripts/sbatch/experiments/mixup_variants/1_train_mixup_static.sh)
echo "Job gestartet: Variante A (Statisch) [ID: $JOB_STATIC]"

JOB_DYNAMIC=$(sbatch --parsable scripts/sbatch/experiments/mixup_variants/1_train_mixup_dynamic.sh)
echo "Job gestartet: Variante B (Dynamisch) [ID: $JOB_DYNAMIC]"

JOB_HYBRID=$(sbatch --parsable scripts/sbatch/experiments/mixup_variants/1_train_mixup_hybrid.sh)
echo "Job gestartet: Variante C (Hybrid) [ID: $JOB_HYBRID]"

JOB_CYCLIC=$(sbatch --parsable scripts/sbatch/experiments/mixup_variants/1_train_mixup_hybrid_cyclic.sh)
echo "Job gestartet: Variante D (Hybrid + Cyclic LR) [ID: $JOB_CYCLIC]"

# 2. Evaluation nach Abschluss aller Trainings-Jobs
JOB_EVAL=$(sbatch --parsable --dependency=afterok:${JOB_STATIC}:${JOB_DYNAMIC}:${JOB_HYBRID}:${JOB_CYCLIC} scripts/sbatch/experiments/mixup_variants/2_evaluate_mixup_variants.sh)
echo "Evaluations-Job eingereiht (abhängig von $JOB_STATIC, $JOB_DYNAMIC, $JOB_HYBRID, $JOB_CYCLIC) [ID: $JOB_EVAL]"

echo "=== Alle Jobs erfolgreich an Slurm übergeben ==="
