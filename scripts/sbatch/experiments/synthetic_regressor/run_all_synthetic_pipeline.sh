#!/bin/bash
# =============================================================================
# Synthetic Regressor Experiment Runner
# =============================================================================

set -e

mkdir -p results/logs data/synthetic results/models/synthetic results/evaluation

echo "=== Submitting Synthetic Regressor Experiment Pipeline ==="

# Stufe 1: Synthetische Daten mit LLM erzeugen
JOB1=$(sbatch --parsable scripts/sbatch/experiments/synthetic_regressor/1_generate_synthetic_steps_lh.sh)
echo "Step 1 (Generate Steps LH): Job ID $JOB1"

JOB2=$(sbatch --parsable scripts/sbatch/experiments/synthetic_regressor/2_generate_synthetic_steps_corpus.sh)
echo "Step 2 (Generate Steps Corpus): Job ID $JOB2"

# Stufe 2: Synthetischen Regressor trainieren
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1:$JOB2 scripts/sbatch/experiments/synthetic_regressor/3_train_synthetic_regressor.sh)
echo "Step 3 (Train Synthetic Regressor): Job ID $JOB3"

# Stufe 3: SFT mit Synthetischem Regressor als Eval
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 scripts/sbatch/experiments/synthetic_regressor/4_train_sft_synthetic.sh)
echo "Step 4 (Train SFT Synthetic): Job ID $JOB4"

# Stufe 4: DPO Paare mit Synthetischem Regressor Reward generieren
JOB5=$(sbatch --parsable --dependency=afterok:$JOB3:$JOB4 scripts/sbatch/experiments/synthetic_regressor/5_generate_dpo_synthetic.sh)
echo "Step 5 (Generate DPO Synthetic): Job ID $JOB5"

# Stufe 5: DPO Training
JOB6=$(sbatch --parsable --dependency=afterok:$JOB4:$JOB5 scripts/sbatch/experiments/synthetic_regressor/6_train_dpo_synthetic.sh)
echo "Step 6 (Train DPO Synthetic): Job ID $JOB6"

# Stufe 6: Evaluierung
JOB7=$(sbatch --parsable --dependency=afterok:$JOB3:$JOB6 scripts/sbatch/experiments/synthetic_regressor/7_evaluate_synthetic_experiments.sh)
echo "Step 7 (Evaluate Synthetic Experiments): Job ID $JOB7"

echo "=== Synthetic Pipeline Submitted Successfully! ==="
