#!/bin/bash
#SBATCH --job-name=13_evaluate_pipeline
#SBATCH --partition=research
#SBATCH --time=02:00:00
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

mkdir -p results/evaluation
mkdir -p results/logs/run_pipeline results/plots/run_pipeline results/evaluation
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU


echo "=== Evaluierung des finalen DPO Modells vs. SFT Baseline auf dem ungesehenen Lebenshilfe Benchmark ==="
date

srun python scripts/evaluation/evaluate_dpo_ladder_model.py \
    --test_data_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --sft_model_path "results/models/sft" \
    --dpo_model_path "results/models/dpo" \
    --base_model_name "facebook/mbart-large-50" \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --output_summary "results/evaluation/pipeline_final_summary.csv" \
    --output_details "results/evaluation/pipeline_final_details.csv" \
    --max_source_len 256 \
    --max_target_len 256 \
    --batch_size 4

echo "=== Pipeline Evaluierung abgeschlossen ==="
date
