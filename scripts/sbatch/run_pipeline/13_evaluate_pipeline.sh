#!/bin/bash
#SBATCH --job-name=13_evaluate_pipeline
#SBATCH --partition=research
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

mkdir -p results/evaluation
mkdir -p results/logs

echo "=== Evaluierung des finalen DPO Modells vs. SFT Baseline auf dem ungesehenen Lebenshilfe Benchmark ==="
date

srun python scripts/evaluation/evaluate_dpo_ladder_model.py \
    --test_data_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --sft_model_path "results/models/sft" \
    --dpo_model_path "results/models/dpo" \
    --base_model_name "facebook/mbart-large-50" \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --sbert_model_name "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" \
    --output_summary "results/evaluation/pipeline_final_summary.csv" \
    --output_details "results/evaluation/pipeline_final_details.csv" \
    --max_source_len 500 \
    --max_target_len 500 \
    --batch_size 4

echo "=== Pipeline Evaluierung abgeschlossen ==="
date
