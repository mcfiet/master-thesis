#!/bin/bash
# ==============================================================================
# Pipeline-Orchestrierung: Klassifikator-Stabilitäts- und Kapazitätsexperiment
# Unterstützt sowohl SLURM (sbatch) als auch direkte Ausführung im Terminal (bash)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

echo "========================================================================"
echo "STARTING CLASSIFIER STABILITY PIPELINE"
echo "Working directory: $(pwd)"
echo "========================================================================"

if command -v sbatch &> /dev/null; then
    echo "[MODE] SLURM erkannt - reiche Jobs ein..."
    JOB_TRAIN=$(sbatch scripts/sbatch/experiments/metric/classifier_stability/1_train_stability_experiment.sh | awk '{print $NF}')
    echo "[SUBMITTED] 1_train_stability_experiment.sh (JobID: $JOB_TRAIN)"

    JOB_EVAL=$(sbatch --dependency=afterok:$JOB_TRAIN scripts/sbatch/experiments/metric/classifier_stability/2_evaluate_stability_experiment.sh | awk '{print $NF}')
    echo "[SUBMITTED] 2_evaluate_stability_experiment.sh (JobID: $JOB_EVAL, depends on $JOB_TRAIN)"
else
    echo "[MODE] Kein SLURM vorhanden - führe sequentiell im lokalen Terminal aus..."
    bash scripts/sbatch/experiments/metric/classifier_stability/1_train_stability_experiment.sh
    bash scripts/sbatch/experiments/metric/classifier_stability/2_evaluate_stability_experiment.sh
fi

echo "========================================================================"
echo "Klassifikator-Stabilitäts-Pipeline erfolgreich abgeschlossen / eingereicht."
echo "========================================================================"
