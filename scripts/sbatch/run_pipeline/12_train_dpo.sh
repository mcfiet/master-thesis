#!/bin/bash
#SBATCH --job-name=12_train_dpo
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/run_pipeline/%x_%j.out
#SBATCH --error=results/logs/run_pipeline/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p results/models/dpo
mkdir -p results/logs/run_pipeline results/plots/run_pipeline results/evaluation
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


echo "=== Start DPO Training (LoRA auf mBART-50 SFT) ==="
date

srun python scripts/modeling/train_dpo.py \
    --model_name_or_path "results/models/sft" \
    --train_file "data/corpus/dpo_pairs_mixup.jsonl" \
    --eval_file "data/corpus/dpo_pairs_mixup_eval.jsonl" \
    --output_dir "results/models/dpo" \
    --loss_type "sum" \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --beta 0.01 \
    --learning_rate 5e-6 \
    --epochs 3 \
    --batch_size 1 \
    --accumulation_steps 16 \
    --patience 3 \
    --max_source_len 1024 \
    --max_target_len 1024

echo "=== DPO Training abgeschlossen ==="
date
