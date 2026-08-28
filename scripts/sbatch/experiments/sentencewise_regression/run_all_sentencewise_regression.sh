#!/bin/bash
# ==============================================================================
# Pipeline-Orchestrierung: Sentence-wise Regressor Experiment
# 1. Training des Satz-Regressors mit Sub-Sentence MixUp
# 2. Umfassende Evaluation & Generierung aller Plots
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

echo "========================================================================"
echo "STARTING SENTENCE-WISE REGRESSION PIPELINE"
echo "Working directory: $(pwd)"
echo "========================================================================"

# Job 1: Training
JOB1=$(sbatch scripts/sbatch/experiments/sentencewise_regression/1_train_sentence_regressor.sh | awk '{print $NF}')
echo "[SUBMITTED] 1_train_sentence_regressor.sh (JobID: $JOB1)"

# Job 2: Evaluation (Depends on Job 1)
JOB2=$(sbatch --dependency=afterok:$JOB1 scripts/sbatch/experiments/sentencewise_regression/2_evaluate_sentencewise_regression.sh | awk '{print $NF}')
echo "[SUBMITTED] 2_evaluate_sentencewise_regression.sh (JobID: $JOB2)"

echo "========================================================================"
echo "Pipeline submitted successfully."
echo "========================================================================"
