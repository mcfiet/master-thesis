#!/bin/bash
# ==============================================================================
# Master Pipeline Runner: Token Length Experiment mit Jina Long-Context Embeddings (8192 Tokens)
# ==============================================================================
# Ablauf:
# 1. DPO-Paar-Generierung mit Jina Embeddings (256, 500, 1000)
# 2. DPO Training (256, 500, 1000)
# 3. Vollständige Evaluation mit Jina Long-Context Metrik
# ==============================================================================

set -e

echo "=== Starte Token-Längen-Experiment mit Jina Embeddings (8192 Tokens) ==="

# 1. Generate DPO Pairs with Jina
JOB_GEN_256=$(sbatch --parsable scripts/sbatch/experiments/token_length_jina/3_generate_dpo_pairs_256_jina.sh)
JOB_GEN_500=$(sbatch --parsable scripts/sbatch/experiments/token_length_jina/3_generate_dpo_pairs_500_jina.sh)
JOB_GEN_1000=$(sbatch --parsable scripts/sbatch/experiments/token_length_jina/3_generate_dpo_pairs_1000_jina.sh)

echo "DPO Generation Jobs eingereicht: $JOB_GEN_256 (256), $JOB_GEN_500 (500), $JOB_GEN_1000 (1000)"

# 2. Train DPO Models
JOB_DPO_256=$(sbatch --parsable --dependency=afterok:$JOB_GEN_256 scripts/sbatch/experiments/token_length_jina/4_train_dpo_256_jina.sh)
JOB_DPO_500=$(sbatch --parsable --dependency=afterok:$JOB_GEN_500 scripts/sbatch/experiments/token_length_jina/4_train_dpo_500_jina.sh)
JOB_DPO_1000=$(sbatch --parsable --dependency=afterok:$JOB_GEN_1000 scripts/sbatch/experiments/token_length_jina/4_train_dpo_1000_jina.sh)

echo "DPO Training Jobs eingereicht (mit Abhängigkeiten): $JOB_DPO_256, $JOB_DPO_500, $JOB_DPO_1000"

# 3. Run Evaluation with Jina Embeddings
JOB_EVAL=$(sbatch --parsable --dependency=afterok:$JOB_DPO_256:$JOB_DPO_500:$JOB_DPO_1000 scripts/sbatch/experiments/token_length_jina/5_run_full_evaluation_jina.sh)

echo "Evaluationsjob eingereicht: $JOB_EVAL (wartet auf alle DPO-Jobs)"
echo "=== Alle Jobs erfolgreich mit SLURM-Abhängigkeiten gestartet! ==="
