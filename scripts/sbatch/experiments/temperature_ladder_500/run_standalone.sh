#!/bin/bash
# =============================================================================
# Standalone Execution Script (No SLURM / No Time Limit)
# =============================================================================
# Runs the full 500-token Temperature Ladder DPO pipeline sequentially:
#   1. Prepares 10kGNAD corpus (uncapped / full dataset if desired)
#   2. Generates DPO pairs with Progressive Temperature Ladder
#   3. Trains DPO model (500 tokens, loss: mean)
#   4. Evaluates on Lebenshilfe benchmark
# =============================================================================

set -e

# Configuration (Edit here if needed)
MAX_SAMPLES="" # Empty string for ALL ~10,000 articles, or e.g. "--max_samples 5000"
BATCH_SIZE=4
CUDA_DEV=0

export CUDA_VISIBLE_DEVICES=$CUDA_DEV
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p data/temperature_ladder_500
mkdir -p results/models/temperature_ladder_500/dpo_w05_w05
mkdir -p results/evaluation
mkdir -p results/logs

echo "========================================================================"
echo "Starting Standalone Temperature Ladder 500 Pipeline (GPU: $CUDA_DEV)"
echo "Started at: $(date)"
echo "========================================================================"

# --- STEP 0: PREPARE CORPUS ---
echo -e "\n[1/4] Preparing 10kGNAD Corpus..."
python scripts/data/prepare_10kgnad_corpus.py \
    --output_file "data/temperature_ladder_500/corpus_10kgnad_len500_as.json" \
    --min_tokens 50 \
    --max_tokens 500 \
    $MAX_SAMPLES

# --- STEP 1: GENERATE DPO PAIRS ---
echo -e "\n[2/4] Generating DPO Preference Pairs (Temperature Ladder)..."
python scripts/modeling/generate_dpo_dataset_ladder.py \
    --corpus_path "data/temperature_ladder_500/corpus_10kgnad_len500_as.json" \
    --sft_model_path "results/models/token_length_exp/sft_len500" \
    --base_model_name "facebook/mbart-large-50" \
    --max_source_len 500 \
    --max_target_len 500 \
    --reward_max_seq_len 500 \
    --reward_model_path "results/models/token_length_exp/bilstm_mixup_regression_500.pt" \
    --reward_vocab_path "data/token_length_exp/mixup_vocab_500.json" \
    --sbert_model_name "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" \
    --temperature_ladder 0.7 0.8 0.9 1.0 \
    --candidates_per_step 3 \
    --max_total_candidates 12 \
    --min_score_margin 0.05 \
    --w_style 0.5 \
    --w_sem 0.5 \
    --batch_size $BATCH_SIZE \
    --output_file "data/temperature_ladder_500/dpo_pairs_w05_w05.jsonl" \
    --val_split_ratio 0.15

# --- STEP 2: TRAIN DPO MODEL ---
echo -e "\n[3/4] Training DPO Model..."
python scripts/modeling/train_dpo.py \
    --model_name_or_path "results/models/token_length_exp/sft_len500" \
    --train_file "data/temperature_ladder_500/dpo_pairs_w05_w05.jsonl" \
    --eval_file "data/temperature_ladder_500/dpo_pairs_w05_w05_eval.jsonl" \
    --output_dir "results/models/temperature_ladder_500/dpo_w05_w05" \
    --loss_type "mean" \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --beta 0.1 \
    --learning_rate 5e-6 \
    --epochs 3 \
    --batch_size 2 \
    --accumulation_steps 8 \
    --patience 3 \
    --max_source_len 500 \
    --max_target_len 500

# --- STEP 3: EVALUATE ---
echo -e "\n[4/4] Evaluating Model on Benchmark..."
python scripts/evaluation/evaluate_dpo_ladder_model.py \
    --test_data_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --sft_model_path "results/models/token_length_exp/sft_len500" \
    --dpo_model_path "results/models/temperature_ladder_500/dpo_w05_w05" \
    --base_model_name "facebook/mbart-large-50" \
    --reward_model_path "results/models/token_length_exp/bilstm_mixup_regression_500.pt" \
    --reward_vocab_path "data/token_length_exp/mixup_vocab_500.json" \
    --sbert_model_name "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" \
    --output_summary "results/evaluation/temperature_ladder_500_summary.csv" \
    --output_details "results/evaluation/temperature_ladder_500_details.csv" \
    --max_source_len 500 \
    --max_target_len 500 \
    --batch_size $BATCH_SIZE

echo -e "\n========================================================================"
echo "Pipeline Completed Successfully at: $(date)"
echo "========================================================================"
