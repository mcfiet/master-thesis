#!/bin/bash
#SBATCH --job-name=eval_classifiers
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/classifier_length/%x_%j.out
#SBATCH --error=results/logs/experiments/classifier_length/%x_%j.err

set -e

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p results/logs/experiments/classifier_length results/plots/experiments/classifier_length results/evaluation

python3 scripts/experiments/evaluate_classifier_length_experiment.py \
    --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --sent_model_path results/models/bilstm_sentence_classifier.pt \
    --sent_vocab_path data/vocabs/sentence_vocab.json \
    --art_256_model results/models/classifier_length_exp/bilstm_article_classifier_256.pt \
    --art_256_vocab data/classifier_length_exp/article_vocab_256.json \
    --art_512_model results/models/classifier_length_exp/bilstm_article_classifier_512.pt \
    --art_512_vocab data/classifier_length_exp/article_vocab_512.json \
    --art_1024_model results/models/classifier_length_exp/bilstm_article_classifier_1024.pt \
    --art_1024_vocab data/classifier_length_exp/article_vocab_1024.json \
    --output_csv results/evaluation/classifier_length_comparison_eval.csv \
    --summary_json results/evaluation/classifier_length_summary.json \
    --plot_dir results/plots/experiments/classifier_length
