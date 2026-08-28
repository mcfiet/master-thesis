#!/bin/bash
# ==============================================================================
# Pipeline-Orchestrierung: Regressor-Längen-Experiment
# Trainiert Satz-Regressor & 256, 512, 1024 Token MixUp-Modelle & führt Evaluation durch
# ==============================================================================

set -e

# Robust zum Repository-Root navigieren
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

echo "========================================================================"
echo "STARTING REGRESSOR LENGTH PIPELINE"
echo "Working directory: $(pwd)"
echo "========================================================================"

JOB_256=$(sbatch scripts/sbatch/experiments/metric/regressor_length/1_train_mixup_regressor_256.sh | awk '{print $NF}')
echo "[SUBMITTED] 1_train_mixup_regressor_256.sh (JobID: $JOB_256)"

JOB_512=$(sbatch scripts/sbatch/experiments/metric/regressor_length/1_train_mixup_regressor_512.sh | awk '{print $NF}')
echo "[SUBMITTED] 1_train_mixup_regressor_512.sh (JobID: $JOB_512)"

JOB_1024=$(sbatch scripts/sbatch/experiments/metric/regressor_length/1_train_mixup_regressor_1024.sh | awk '{print $NF}')
echo "[SUBMITTED] 1_train_mixup_regressor_1024.sh (JobID: $JOB_1024)"

JOB_EVAL=$(sbatch --dependency=afterok:$JOB_256,$JOB_512,$JOB_1024 scripts/sbatch/experiments/metric/regressor_length/2_evaluate_regressors.sh | awk '{print $NF}')
echo "[SUBMITTED] 2_evaluate_regressors.sh (JobID: $JOB_EVAL, depends on $JOB_256, $JOB_512, $JOB_1024)"

echo "========================================================================"
echo "Regressor-Pipeline erfolgreich eingereicht."
echo "========================================================================"
