#!/bin/bash
#SBATCH --job-name=1_scaling_mixtures
#SBATCH --partition=research
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_48gb:1
#SBATCH --output=results/logs/experiments/data_scaling/%x_%j.out
#SBATCH --error=results/logs/experiments/data_scaling/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi



echo "=== Starting MixUp Multiplier Scaling Grid (mixtures_per_pair) ==="
mkdir -p results/logs/experiments/data_scaling results/plots/experiments/data_scaling results/evaluation data/data_scaling

MIXTURE_VALUES=(2 5 10 20 40 80 160 320)

for M in "${MIXTURE_VALUES[@]}"; do
    EXP_NAME="scale_mixtures_m${M}"
    echo "--- Running Experiment: ${EXP_NAME} (mixtures_per_pair=${M}) ---"
    
    srun python scripts/experiments/data_scaling/train_mixup_scaling.py \
        --csv_path data/analysis/corpus_master.csv \
        --min_sim 0.80 \
        --max_sim 0.98 \
        --max_seq_len 1024 \
        --batch_size 64 \
        --epochs 80 \
        --lr 0.001 \
        --mixtures_per_pair ${M} \
        --train_fraction 1.0 \
        --experiment_group "mixtures_scaling" \
        --experiment_name "${EXP_NAME}" \
        --output_dir results/experiments/data_scaling \
        --vocab_save_path data/data_scaling/mixup_vocab.json
done

echo "=== MixUp Multiplier Scaling Grid Completed ==="
