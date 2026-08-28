#!/bin/bash
#SBATCH --job-name=eval_sentencewise_regression
#SBATCH --output=results/logs/experiments/sentencewise_regression/%x_%j.out
#SBATCH --error=results/logs/experiments/sentencewise_regression/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:mig_24gb:1

# Environment aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p results/logs/experiments/sentencewise_regression
mkdir -p results/plots
mkdir -p results/evaluation

python3 scripts/experiments/evaluate_sentencewise_regression.py \
    --dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --sent_reg_model results/models/bilstm_sentence_regressor.pt \
    --sent_reg_vocab data/vocabs/sentence_regressor_vocab.json \
    --mixup_reg_model results/models/bilstm_mixup_regression.pt \
    --mixup_reg_vocab data/vocabs/mixup_vocab.json \
    --output_csv results/evaluation/sentencewise_regression_lh_eval.csv \
    --unified_csv results/evaluation/unified_lh_benchmark_eval.csv \
    --summary_json results/evaluation/sentencewise_regression_summary.json \
    --plot_dir results/plots
