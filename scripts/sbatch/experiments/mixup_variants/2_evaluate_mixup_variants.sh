#!/bin/bash
#SBATCH --job-name=eval_mixup_variants
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/mixup_variants/%x_%j.out
#SBATCH --error=results/logs/experiments/mixup_variants/%x_%j.err

set -e

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p results/logs/experiments/mixup_variants results/plots/experiments/mixup_variants results/evaluation
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

echo "=== Starte Evaluation: MixUp Modell-Varianten Vergleich ==="
srun python scripts/evaluation/evaluate_mixup_variants.py \
    --corpus_csv data/analysis/corpus_master.csv \
    --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --vocab_path data/mixup_variants/mixup_vocab.json \
    --model_static results/models/mixup_variants/bilstm_mixup_regression_static.pt \
    --model_dynamic results/models/mixup_variants/bilstm_mixup_regression_dynamic.pt \
    --model_hybrid results/models/mixup_variants/bilstm_mixup_regression_hybrid.pt \
    --model_hybrid_cyclic results/models/mixup_variants/bilstm_mixup_regression_hybrid_cyclic.pt \
    --output_csv results/evaluation/mixup_variants_eval.csv \
    --output_dir results/evaluation \
    --plot_dir results/plots/experiments/mixup_variants \
    --max_seq_len 150 \
    --mixtures_per_pair 10

echo "=== Evaluation der MixUp-Varianten abgeschlossen ==="
