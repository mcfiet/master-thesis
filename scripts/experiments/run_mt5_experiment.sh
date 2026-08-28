#!/bin/bash
# ==============================================================================
# Runner Script: Google mT5-base Experiment Pipeline (SFT, DPO, Benchmark)
# ==============================================================================
# Unterstützt modulare Ausführung:
#   Option A (Nur SFT + Eval):
#     bash scripts/experiments/run_mt5_experiment.sh --sft-only
#   Option B (Volle Pipeline: SFT -> DPO -> Benchmark):
#     bash scripts/experiments/run_mt5_experiment.sh --all
#   Einzelne Schritte:
#     bash scripts/experiments/run_mt5_experiment.sh --dpo-only
#     bash scripts/experiments/run_mt5_experiment.sh --eval-only
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Python Binary: bevorzugt .venv falls vorhanden
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

# Standard-Konfiguration
MODEL_NAME="google/mt5-base"
PROMPT_PREFIX="Vereinfache zu Leichter Sprache: "
MAX_SOURCE_LEN=1024
MAX_TARGET_LEN=1024
BATCH_SIZE=1
ACCUMULATION_STEPS=16
EPOCHS_SFT=10
EPOCHS_DPO=3
LR_SFT=5e-5
LR_DPO=5e-6
BETA_DPO=0.10

CORPUS_PATH="data/analysis/corpus_master.csv"
LEBENSHILFE_PATH="data/lebenshilfe/lebenshilfe_dataset_clean.json"
DPO_TRAIN_FILE="data/corpus/dpo_pairs_mixup.jsonl"
DPO_EVAL_FILE="data/corpus/dpo_pairs_mixup_eval.jsonl"
REWARD_MODEL_PATH="results/models/token_length_exp/bilstm_mixup_regression_1024.pt"
REWARD_VOCAB_PATH="data/token_length_exp/mixup_vocab_1024.json"

if [ ! -f "$REWARD_MODEL_PATH" ] && [ -f "results/models/bilstm_mixup_regression.pt" ]; then
    REWARD_MODEL_PATH="results/models/bilstm_mixup_regression.pt"
    REWARD_VOCAB_PATH="data/vocabs/mixup_vocab.json"
fi

SFT_OUTPUT_DIR="results/models/mt5_exp/sft_mt5_base"
DPO_OUTPUT_DIR="results/models/mt5_exp/dpo_mt5_base"
LOG_DIR="results/logs/experiments/mt5_exp"
PLOT_DIR="results/plots/experiments/mt5_exp"
EVAL_DIR="results/evaluation"

mkdir -p "$LOG_DIR" "$PLOT_DIR" "$EVAL_DIR" "results/models/mt5_exp"

# Modus parsen
MODE="${1:---all}"

echo "========================================================================"
echo " Google mT5-base Experiment Pipeline"
echo " Startzeit:     $(date)"
echo " Modus:         $MODE"
echo " Basismodell:   $MODEL_NAME"
echo " Python:        $PYTHON_BIN"
echo " Arbeitsordner: $REPO_ROOT"
echo "========================================================================"

# ==============================================================================
# SCHRITT 1: SFT TRAINING (OPTION A / TEIL 1 VON OPTION B)
# ==============================================================================
if [ "$MODE" == "--all" ] || [ "$MODE" == "--sft-only" ] || [ "$MODE" == "-sft" ]; then
    echo ""
    echo ">>> [1/3] Starte mT5-base SFT Training (Supervised Fine-Tuning)..."
    echo "    Output Directory: $SFT_OUTPUT_DIR"
    
    "$PYTHON_BIN" scripts/modeling/train_sft.py \
        --corpus_path "$CORPUS_PATH" \
        --lh_dataset_path "$LEBENSHILFE_PATH" \
        --output_dir "$SFT_OUTPUT_DIR" \
        --log_dir "$LOG_DIR" \
        --plot_dir "$PLOT_DIR" \
        --model_name "$MODEL_NAME" \
        --prompt_prefix "$PROMPT_PREFIX" \
        --min_sim 0.70 \
        --max_sim 1.0 \
        --max_source_len $MAX_SOURCE_LEN \
        --max_target_len $MAX_TARGET_LEN \
        --batch_size $BATCH_SIZE \
        --accumulation_steps $ACCUMULATION_STEPS \
        --epochs $EPOCHS_SFT \
        --lr $LR_SFT \
        --use_peft \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --reward_model_path "$REWARD_MODEL_PATH" \
        --reward_vocab_path "$REWARD_VOCAB_PATH" \
        --reward_max_seq_len 1024
    
    echo ">>> [ERFOLG] mT5-base SFT Training abgeschlossen!"
fi

# ==============================================================================
# SCHRITT 2: DPO TRAINING (OPTION B - TEIL 2)
# ==============================================================================
if [ "$MODE" == "--all" ] || [ "$MODE" == "--dpo-only" ] || [ "$MODE" == "-dpo" ]; then
    echo ""
    echo ">>> [2/3] Starte mT5-base DPO Training (Direct Preference Optimization)..."
    echo "    SFT Input Model:  $SFT_OUTPUT_DIR"
    echo "    DPO Output Model: $DPO_OUTPUT_DIR"
    
    if [ ! -d "$SFT_OUTPUT_DIR" ]; then
        echo "FEHLER: SFT-Modell unter '$SFT_OUTPUT_DIR' nicht gefunden! Bitte zuerst SFT trainieren."
        exit 1
    fi
    
    if [ ! -f "$DPO_TRAIN_FILE" ]; then
        echo "Warnung: '$DPO_TRAIN_FILE' nicht gefunden, prüfe Fallback..."
        if [ -f "data/dpo/dpo_preference_pairs_synthetic.jsonl" ]; then
            DPO_TRAIN_FILE="data/dpo/dpo_preference_pairs_synthetic.jsonl"
            DPO_EVAL_FILE="data/dpo/dpo_preference_pairs_synthetic_eval.jsonl"
        fi
    fi
    
    "$PYTHON_BIN" scripts/modeling/train_dpo.py \
        --sft_model_path "$SFT_OUTPUT_DIR" \
        --train_file "$DPO_TRAIN_FILE" \
        --eval_file "$DPO_EVAL_FILE" \
        --output_dir "$DPO_OUTPUT_DIR" \
        --log_dir "$LOG_DIR" \
        --plot_dir "$PLOT_DIR" \
        --prompt_prefix "$PROMPT_PREFIX" \
        --max_source_len $MAX_SOURCE_LEN \
        --max_target_len $MAX_TARGET_LEN \
        --batch_size $BATCH_SIZE \
        --accumulation_steps $ACCUMULATION_STEPS \
        --epochs $EPOCHS_DPO \
        --beta $BETA_DPO \
        --lr $LR_DPO \
        --use_peft \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05
    
    echo ">>> [ERFOLG] mT5-base DPO Training abgeschlossen!"
fi

# ==============================================================================
# SCHRITT 3: BENCHMARK & EVALUATION
# ==============================================================================
if [ "$MODE" == "--all" ] || [ "$MODE" == "--eval-only" ] || [ "$MODE" == "--sft-only" ] || [ "$MODE" == "-eval" ]; then
    echo ""
    echo ">>> [3/3] Starte quantitative Benchmark-Evaluation (mBART-50 vs. mT5-base)..."
    
    "$PYTHON_BIN" scripts/evaluation/evaluate_token_length_experiment.py \
        --test_data_path "$LEBENSHILFE_PATH" \
        --reward_model_path "$REWARD_MODEL_PATH" \
        --reward_vocab_path "$REWARD_VOCAB_PATH" \
        --output_summary "$EVAL_DIR/mt5_vs_mbart_comparison_summary.csv" \
        --output_details "$EVAL_DIR/mt5_vs_mbart_comparison_detailed.csv" || true

    echo ">>> [ERFOLG] Evaluation abgeschlossen! Ergebnisse unter: $EVAL_DIR/mt5_vs_mbart_comparison_summary.csv"
fi

echo ""
echo "========================================================================"
echo " Pipeline erfolgreich abgeschlossen!"
echo " Endzeit: $(date)"
echo "========================================================================"
