#!/bin/bash
# ==============================================================================
# Standalone Runner: DPO Beta Sweep Experiment (Beta = 0.01, 0.05, 0.10, 0.20, 0.50)
# Works on any machine with GPU (without SLURM / sbatch).
# Automatically detects available GPUs and runs trainings either in parallel or sequentially.
# ==============================================================================

set -e

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/.venv/bin/activate" ]; then
    source "$HOME/.venv/bin/activate"
fi

mkdir -p results/models/dpo_beta_sweep
mkdir -p results/logs/experiments/dpo_beta_sweep
mkdir -p results/plots/experiments/dpo_beta_sweep
mkdir -p results/evaluation

echo "================================================================================"
echo "Starting DPO Beta Sweep (Standalone Mode)"
echo "================================================================================"

# Check for CUDA GPU
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo "Detected $NUM_GPUS NVIDIA GPU(s):"
    nvidia-smi -L
else
    NUM_GPUS=1
    echo "nvidia-smi not found, defaulting to single GPU/CPU."
fi

TRAIN_FILE="data/corpus/dpo_pairs_mixup.jsonl"
EVAL_FILE="data/corpus/dpo_pairs_mixup_eval.jsonl"
SFT_MODEL="results/models/sft"

# Verify prerequisites
if [ ! -d "$SFT_MODEL" ]; then
    echo "[FAIL] Error: SFT model directory not found at '$SFT_MODEL'."
    echo "Please ensure the SFT checkpoint is copied to '$SFT_MODEL'."
    exit 1
fi

if [ ! -f "$TRAIN_FILE" ]; then
    echo "[FAIL] Error: Training dataset not found at '$TRAIN_FILE'."
    exit 1
fi

BETAS=("0.01" "0.05" "0.10" "0.20" "0.50")
NAMES=("dpo_beta_001" "dpo_beta_005" "dpo_beta_010" "dpo_beta_020" "dpo_beta_050")

train_single_beta() {
    local BETA=$1
    local NAME=$2
    local GPU_ID=$3

    echo "[Start] Training Beta = $BETA ($NAME) on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/modeling/train_dpo.py \
        --model_name_or_path "$SFT_MODEL" \
        --train_file "$TRAIN_FILE" \
        --eval_file "$EVAL_FILE" \
        --output_dir "results/models/dpo_beta_sweep/$NAME" \
        --loss_type "sum" \
        --use_peft \
        --lora_r 16 \
        --lora_alpha 32 \
        --beta "$BETA" \
        --learning_rate 5e-6 \
        --epochs 3 \
        --batch_size 2 \
        --accumulation_steps 8 \
        --patience 3 \
        --max_source_len 256 \
        --max_target_len 256 \
        --log_dir "results/logs/experiments/dpo_beta_sweep" \
        > "results/logs/experiments/dpo_beta_sweep/train_${NAME}.out" 2>&1
    echo "[OK] [Done] Training Beta = $BETA ($NAME) finished successfully!"
}

# If multiple GPUs are available (>= 5), run all in parallel
if [ "$NUM_GPUS" -ge 5 ]; then
    echo "Multi-GPU mode: Running all 5 Beta trainings concurrently across 5 GPUs..."
    for i in "${!BETAS[@]}"; do
        train_single_beta "${BETAS[$i]}" "${NAMES[$i]}" "$i" &
    done
    wait
elif [ "$NUM_GPUS" -ge 2 ]; then
    echo "Multi-GPU mode: Dispatching across $NUM_GPUS GPUs..."
    for i in "${!BETAS[@]}"; do
        GPU_ID=$((i % NUM_GPUS))
        train_single_beta "${BETAS[$i]}" "${NAMES[$i]}" "$GPU_ID" &
        if [ $(( (i + 1) % NUM_GPUS )) -eq 0 ]; then
            wait
        fi
    done
    wait
else
    echo "Single-GPU mode: Running 5 Beta trainings sequentially..."
    for i in "${!BETAS[@]}"; do
        train_single_beta "${BETAS[$i]}" "${NAMES[$i]}" "0"
    done
fi

echo "================================================================================"
echo "All 5 DPO Models trained successfully! Starting full evaluation..."
echo "================================================================================"

python scripts/evaluation/evaluate_dpo_beta_experiment.py \
    --test_data_path "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --sft_model_path "$SFT_MODEL" \
    --beta_models_dir "results/models/dpo_beta_sweep" \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --output_summary_csv "results/evaluation/dpo_beta_comparison_summary.csv" \
    --output_details_csv "results/evaluation/dpo_beta_comparison_details.csv" \
    --output_plot_path "results/plots/experiments/dpo_beta_sweep/dpo_beta_pareto_tradeoff.png" \
    --max_source_len 256 \
    --max_target_len 256 \
    --batch_size 4

echo "================================================================================"
echo "DPO Beta Sweep Pipeline completely finished!"
echo "Summary: results/evaluation/dpo_beta_comparison_summary.csv"
echo "Details: results/evaluation/dpo_beta_comparison_details.csv"
echo "================================================================================"
