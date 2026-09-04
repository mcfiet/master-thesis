#!/bin/bash
#SBATCH --job-name=eval_synthetic_rules_256
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_48gb:1
#SBATCH --output=results/logs/experiments/rule_sensitivity/%x_%j.out
#SBATCH --error=results/logs/experiments/rule_sensitivity/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
elif [ -f "$HOME/master-old/.venv/bin/activate" ]; then
    source "$HOME/master-old/.venv/bin/activate"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p results/logs/experiments/rule_sensitivity          results/plots/experiments/rule_sensitivity          results/evaluation          data/experiments/rule_sensitivity          thesis/images

echo "=== Starte Synthetischen 256-Token Regel-Benchmark ==="
date

echo "--- 1. Erzeuge synthetische Benchmark-Texte (100–256 Tokens) ---"
python scripts/data/generate_synthetic_rule_benchmark.py     --output_json "data/experiments/rule_sensitivity/synthetic_rule_benchmark_256.json"     --output_csv "data/experiments/rule_sensitivity/synthetic_rule_benchmark_256.csv"

echo "--- 2. Führe Inferenz (256-Token Regressor & klassische Metriken) durch ---"
python scripts/evaluation/evaluate_synthetic_rule_benchmark.py     --input_json "data/experiments/rule_sensitivity/synthetic_rule_benchmark_256.json"     --output_csv "results/evaluation/synthetic_rule_benchmark_256_eval.csv"     --summary_json "results/evaluation/synthetic_rule_benchmark_256_summary.json"     --model_path "results/models/regressor_length_exp/bilstm_mixup_regression_256.pt"     --vocab_path "data/regressor_length_exp/mixup_vocab_256.json"

echo "--- 3. Generiere Visualisierungen ---"
python scripts/visualization/plot_synthetic_rule_benchmark.py     --eval_csv "results/evaluation/synthetic_rule_benchmark_256_eval.csv"     --summary_json "results/evaluation/synthetic_rule_benchmark_256_summary.json"     --plot_dir "results/plots/experiments/rule_sensitivity"     --thesis_dir "thesis/images"

echo "=== Benchmark erfolgreich abgeschlossen! ==="
date
