#!/bin/bash
#SBATCH --job-name=eval_regressors
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/regressor_length/%x_%j.out
#SBATCH --error=results/logs/experiments/regressor_length/%x_%j.err

set -e

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p results/logs/experiments/regressor_length results/plots/experiments/regressor_length results/evaluation

python3 scripts/experiments/evaluate_regressor_length_experiment.py \
    --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --mix_256_model results/models/regressor_length_exp/bilstm_mixup_regression_256.pt \
    --mix_256_vocab data/regressor_length_exp/mixup_vocab_256.json \
    --mix_512_model results/models/regressor_length_exp/bilstm_mixup_regression_512.pt \
    --mix_512_vocab data/regressor_length_exp/mixup_vocab_512.json \
    --mix_1024_model results/models/regressor_length_exp/bilstm_mixup_regression_1024.pt \
    --mix_1024_vocab data/regressor_length_exp/mixup_vocab_1024.json \
    --output_csv results/evaluation/regressor_length_comparison_eval.csv \
    --summary_json results/evaluation/regressor_length_summary.json \
    --plot_dir results/plots/experiments/regressor_length
