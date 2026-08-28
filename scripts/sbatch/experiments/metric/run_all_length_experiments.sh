#!/bin/bash
# ==============================================================================
# Master-Runner: Startet BEIDE Experiment-Säulen auf dem Cluster
# 1. Klassifikatoren: 256, 512, 1024 Tokens + Gesamtevaluation
# 2. Regressoren: Satz-Regressor + 256, 512, 1024 Tokens MixUp + Gesamtevaluation
# ==============================================================================

set -e

# Robust zum Repository-Root navigieren
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

echo "========================================================================"
echo "STARTING ALL CLASSIFIER AND REGRESSOR LENGTH EXPERIMENTS"
echo "Working directory: $(pwd)"
echo "========================================================================"

echo ""
echo "--- [1/2] KLASSIFIKATOR-EXPERIMENTE EINREICHEN ---"
bash scripts/sbatch/experiments/metric/classifier_length/run_all_classifier_experiments.sh

echo ""
echo "--- [2/2] REGRESSOR-EXPERIMENTE EINREICHEN ---"
bash scripts/sbatch/experiments/metric/regressor_length/run_all_regressor_experiments.sh

echo ""
echo "========================================================================"
echo "Alle Experimente erfolgreich auf dem Cluster eingereicht!"
echo "Überprüfen Sie den Status mit: squeue -u \$USER"
echo "========================================================================"
