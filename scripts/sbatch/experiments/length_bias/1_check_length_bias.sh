#!/bin/bash
set -e
#SBATCH --job-name=eval_length_bias
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_48gb:1
#SBATCH --output=results/logs/experiments/length_bias/%x_%j.out
#SBATCH --error=results/logs/experiments/length_bias/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/logs/experiments/length_bias results/plots/experiments/length_bias results/evaluation

echo "=== Starte Length-Bias & Shortcut Evaluation ==="
date

srun python scripts/evaluation/check_length_bias.py \
    --dataset_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --model_path "results/models/bilstm_article_classifier.pt" \
    --vocab_path "data/vocabs/article_vocab.json" \
    --vocab_source_csv "data/analysis/corpus_master.csv" \
    --output_csv "results/evaluation/length_bias_results.csv"

echo "=== Length-Bias Evaluation erfolgreich abgeschlossen ==="
date
