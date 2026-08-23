#!/bin/bash
#SBATCH --job-name=1_scaling_mixtures
#SBATCH --partition=research
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/data_scaling_mixtures_%j.log

set -e

echo "=== Starting MixUp Multiplier Scaling Grid (mixtures_per_pair) ==="
mkdir -p results/experiments/data_scaling results/logs data/vocabs

MIXTURE_VALUES=(2 5 10 20 40 80)

for M in "${MIXTURE_VALUES[@]}"; do
    EXP_NAME="scale_mixtures_m${M}"
    echo "--- Running Experiment: ${EXP_NAME} (mixtures_per_pair=${M}) ---"
    
    srun python scripts/experiments/data_scaling/train_mixup_scaling.py \
        --csv_path data/analysis/corpus_master.csv \
        --min_sim 0.80 \
        --max_sim 0.98 \
        --max_seq_len 256 \
        --batch_size 64 \
        --epochs 40 \
        --lr 0.001 \
        --mixtures_per_pair ${M} \
        --train_fraction 1.0 \
        --experiment_group "mixtures_scaling" \
        --experiment_name "${EXP_NAME}" \
        --output_dir results/experiments/data_scaling \
        --vocab_save_path data/vocabs/mixup_vocab.json
done

echo "=== MixUp Multiplier Scaling Grid Completed ==="
