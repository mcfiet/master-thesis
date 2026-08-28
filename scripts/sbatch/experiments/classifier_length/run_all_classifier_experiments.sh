#!/bin/bash
# ==============================================================================
# Pipeline-Orchestrierung: Klassifikator-Längen-Experiment
# Trainiert 256, 512, 1024 Token Artikel-Modelle & führt Gesamtevaluation durch
# ==============================================================================

set -e

# Robust zum Repository-Root navigieren
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "========================================================================"
echo "STARTING CLASSIFIER LENGTH PIPELINE"
echo "Working directory: $(pwd)"
echo "========================================================================"

JOB_256=$(sbatch scripts/sbatch/experiments/classifier_length/1_train_article_classifier_256.sh | awk '{print $NF}')
echo "[SUBMITTED] 1_train_article_classifier_256.sh (JobID: $JOB_256)"

JOB_512=$(sbatch scripts/sbatch/experiments/classifier_length/1_train_article_classifier_512.sh | awk '{print $NF}')
echo "[SUBMITTED] 1_train_article_classifier_512.sh (JobID: $JOB_512)"

JOB_1024=$(sbatch scripts/sbatch/experiments/classifier_length/1_train_article_classifier_1024.sh | awk '{print $NF}')
echo "[SUBMITTED] 1_train_article_classifier_1024.sh (JobID: $JOB_1024)"

JOB_EVAL=$(sbatch --dependency=afterok:$JOB_256,$JOB_512,$JOB_1024 scripts/sbatch/experiments/classifier_length/2_evaluate_classifiers.sh | awk '{print $NF}')
echo "[SUBMITTED] 2_evaluate_classifiers.sh (JobID: $JOB_EVAL, depends on $JOB_256, $JOB_512, $JOB_1024)"

echo "========================================================================"
echo "Klassifikator-Pipeline erfolgreich eingereicht."
echo "========================================================================"
