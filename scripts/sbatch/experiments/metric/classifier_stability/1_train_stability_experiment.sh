#!/bin/bash
#SBATCH --job-name=train_clf_stability
#SBATCH --partition=research
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/classifier_stability/%x_%j.out
#SBATCH --error=results/logs/experiments/classifier_stability/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p results/logs/experiments/classifier_stability results/plots/experiments/classifier_stability results/evaluation/classifier_stability
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

echo "========================================================================"
echo "STARTING CLASSIFIER STABILITY TRAINING (MULTI-SEED & CAPACITY)"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "========================================================================"

# Führt das Multi-Seed- und Epochen-Tracking-Training aus
python scripts/experiments/classifier_stability/train_and_track_stability.py \
    --csv_path data/analysis/corpus_master.csv \
    --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --seeds 42 123 456 789 1024 \
    --epochs 30 \
    --batch_size 32 \
    --lr 0.001 \
    --min_sim 0.80 \
    --max_sim 0.98 \
    --output_dir results/evaluation/classifier_stability

echo "========================================================================"
echo "TRAINING COMPLETE."
echo "========================================================================"
