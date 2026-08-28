#!/bin/bash
#SBATCH --job-name=eval_rule_adherence
#SBATCH --partition=research
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=results/logs/experiments/rule_adherence/%x_%j.out
#SBATCH --error=results/logs/experiments/rule_adherence/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


mkdir -p results/logs/experiments/rule_adherence results/plots/experiments/rule_adherence results/evaluation

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
