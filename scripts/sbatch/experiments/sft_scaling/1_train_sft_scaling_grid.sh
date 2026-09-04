#!/bin/bash
#SBATCH --job-name=1_sft_scaling_grid
#SBATCH --partition=research
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/sft_scaling/%x_%j.out
#SBATCH --error=results/logs/experiments/sft_scaling/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi



echo "=== Starte SFT Data Scaling Grid (Fractions: 10%, 25%, 50%, 75%, 100%) ==="
mkdir -p results/logs/experiments/sft_scaling results/plots/experiments/sft_scaling results/evaluation

FRACTION_VALUES=(0.10 0.25 0.50 0.75 1.00)

for F in "${FRACTION_VALUES[@]}"; do
    EXP_NAME="sft_scale_f$(echo "${F}" | sed 's/\.//')"
    echo "=========================================================================="
    echo "--- Starte Experiment: ${EXP_NAME} (train_fraction=${F}) ---"
    echo "=========================================================================="
    
    srun python scripts/experiments/sft_scaling/train_sft_scaling.py \
        --corpus_path data/analysis/corpus_master.csv \
        --test_file data/lebenshilfe/lebenshilfe_dataset_clean.json \
        --output_dir results/experiments/sft_scaling \
        --base_model_name facebook/mbart-large-50 \
        --reward_model_path results/models/bilstm_mixup_regression.pt \
        --reward_vocab_path data/vocabs/mixup_vocab.json \
        --reward_max_seq_len 1024 \
        --sbert_model_name jinaai/jina-embeddings-v2-base-de \
        --train_fraction ${F} \
        --experiment_name "${EXP_NAME}" \
        --min_sim 0.70 \
        --max_sim 0.98 \
        --max_source_len 1024 \
        --max_target_len 1024 \
        --batch_size 4 \
        --accumulation_steps 4 \
        --epochs 30 \
        --lr 1e-4 \
        --patience 10 \
        --seed 42 \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05
done

echo "=== SFT Data Scaling Grid Training erfolgreich beendet! ==="
