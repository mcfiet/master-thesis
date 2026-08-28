#!/bin/bash
set -e
#SBATCH --job-name=10_train_sft
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/run_pipeline/%x_%j.out
#SBATCH --error=results/logs/run_pipeline/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p results/models/sft
mkdir -p results/logs/run_pipeline results/plots/run_pipeline results/evaluation
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU


echo "=== Start SFT Training (mBART-50 auf corpus_master) ==="
date

srun python scripts/modeling/train_sft.py \
    --corpus_path "data/analysis/corpus_master.json" \
    --lh_dataset_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --output_dir "results/models/sft" \
    --min_sim 0.70 \
    --max_sim 0.98 \
    --max_source_len 1024 \
    --max_target_len 1024 \
    --model_name "facebook/mbart-large-50" \
    --batch_size 2 \
    --accumulation_steps 8 \
    --epochs 30 \
    --lr 1e-4 \
    --warmup_ratio 0.10 \
    --patience 10 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --reward_max_seq_len 1024

echo "=== SFT Training abgeschlossen ==="
date
