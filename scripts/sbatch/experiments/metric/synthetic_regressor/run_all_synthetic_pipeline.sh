#!/bin/bash
#SBATCH --job-name=run_all_synthetic_pipeline
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/experiments/synthetic_regressor/%x_%j.out
#SBATCH --error=results/logs/experiments/synthetic_regressor/%x_%j.err

set -e
# =============================================================================
# Synthetic Regressor Experiment Runner
# =============================================================================


mkdir -p results/logs/experiments/synthetic_regressor data/synthetic results/models/experiments/synthetic_regressor results/evaluation

echo "=== Submitting Synthetic Regressor Experiment Pipeline ==="

# Stufe 1: Synthetische Daten mit LLM erzeugen
JOB1=$(sbatch --parsable scripts/sbatch/experiments/metric/synthetic_regressor/1_generate_synthetic_steps_lh.sh)
echo "Step 1 (Generate Steps LH): Job ID $JOB1"

JOB2=$(sbatch --parsable scripts/sbatch/experiments/metric/synthetic_regressor/2_generate_synthetic_steps_corpus.sh)
echo "Step 2 (Generate Steps Corpus): Job ID $JOB2"

# Stufe 2: Synthetischen Regressor trainieren
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1:$JOB2 scripts/sbatch/experiments/metric/synthetic_regressor/3_train_synthetic_regressor.sh)
echo "Step 3 (Train Synthetic Regressor): Job ID $JOB3"

# Stufe 3: Evaluierung (MixUp vs. Synthetic Regressor Unbiased & KDE)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 scripts/sbatch/experiments/metric/synthetic_regressor/4_evaluate_synthetic_experiments.sh)
echo "Step 4 (Evaluate MixUp vs. Synthetic Steps): Job ID $JOB4"

JOB5=$(sbatch --parsable --dependency=afterok:$JOB3 scripts/sbatch/experiments/metric/synthetic_regressor/5_evaluate_synthetic_kde.sh)
echo "Step 5 (Evaluate Synthetic KDE): Job ID $JOB5"

echo "=== Synthetic Pipeline Submitted Successfully! ==="
