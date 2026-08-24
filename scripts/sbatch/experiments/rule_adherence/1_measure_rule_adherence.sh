#!/bin/bash
#SBATCH --job-name=eval_rule_adherence
#SBATCH --partition=research
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

mkdir -p data/analysis results/logs

echo "=== Starte Quantitative Regel-Adhärenz Evaluation ==="
date

srun python scripts/evaluation/measure_rule_adherence.py \
    --mode all \
    --corpus_csv "data/analysis/corpus_master.csv" \
    --models_csv "results/evaluation/token_length_comparison_detailed.csv" \
    --output_corpus "data/analysis/rule_adherence_corpus.csv" \
    --output_models "data/analysis/rule_adherence_ladder_sft.csv"

echo "=== Regel-Adhärenz Evaluation erfolgreich abgeschlossen ==="
date
