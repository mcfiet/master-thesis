#!/bin/bash
set -e
#SBATCH --job-name=eval_synthetic_kde
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_48gb:1
#SBATCH --output=results/logs/experiments/synthetic_regressor/%x_%j.out
#SBATCH --error=results/logs/experiments/synthetic_regressor/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/logs/experiments/synthetic_regressor results/plots/experiments/synthetic_regressor results/evaluation

echo "=== Starte MixUp vs. Synthetic KDE Evaluation auf Lebenshilfe ==="
date

srun python scripts/evaluation/evaluate_mixup_synthetic_kde.py \
    --test_data_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --mixup_model_path "results/models/bilstm_mixup_regression.pt" \
    --mixup_vocab_path "data/vocabs/mixup_vocab.json" \
    --synthetic_model_path "results/models/bilstm_synthetic_regression.pt" \
    --synthetic_vocab_path "data/vocabs/synthetic_vocab.json" \
    --output_csv "results/evaluation/mixup_synthetic_kde_eval.csv"

echo "=== MixUp vs. Synthetic KDE Evaluation erfolgreich abgeschlossen ==="
date
