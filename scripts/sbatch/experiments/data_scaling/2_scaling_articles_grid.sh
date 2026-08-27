#!/bin/bash
#SBATCH --job-name=2_scaling_articles
#SBATCH --partition=research
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/experiments/data_scaling/%x_%j.out
#SBATCH --error=results/logs/experiments/data_scaling/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


set -e

echo "=== Starting Base Article Scaling Grid (train_fraction) ==="
mkdir -p results/logs/experiments/data_scaling results/plots/experiments/data_scaling results/evaluation

FRACTION_VALUES=(0.10 0.25 0.50 0.75 1.00)

for F in "${FRACTION_VALUES[@]}"; do
    EXP_NAME="scale_pairs_f$(echo "${F}" | sed 's/\.//')"
    echo "--- Running Experiment: ${EXP_NAME} (train_fraction=${F}) ---"
    
    srun python scripts/experiments/data_scaling/train_mixup_scaling.py \
        --csv_path data/analysis/corpus_master.csv \
        --min_sim 0.80 \
        --max_sim 0.98 \
        --max_seq_len 256 \
        --batch_size 64 \
        --epochs 40 \
        --lr 0.001 \
        --mixtures_per_pair 20 \
        --train_fraction ${F} \
        --experiment_group "pairs_scaling" \
        --experiment_name "${EXP_NAME}" \
        --output_dir results/experiments/data_scaling \
        --vocab_save_path data/vocabs/mixup_vocab.json
done

echo "=== Base Article Scaling Grid Completed ==="
