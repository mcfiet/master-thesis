#!/bin/bash
#SBATCH --job-name=build_expert_eval_dataset
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/expert_eval/%x_%j.out
#SBATCH --error=results/logs/expert_eval/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p results/logs/expert_eval data/expert_eval results/expert_eval
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

echo "=== Starte Generierung des 50-Item Experten-Evaluationspools auf GPU ==="
date

srun python scripts/evaluation/build_expert_evaluation_set.py \
    --corpus_csv "data/analysis/corpus_master.csv" \
    --lh_json "data/lebenshilfe/lebenshilfe_dataset_clean.json" \
    --sft_model_path "results/models/token_length_exp/sft_len1024" \
    --dpo_model_path "results/models/token_length_exp/dpo_len1024" \
    --reward_model_path "results/models/regressor_length_exp/bilstm_mixup_regression_1024.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --output_dir "data/expert_eval" \
    --seed 42 \
    --device "cuda"

echo "=== Experten-Evaluationspool erfolgreich erstellt ==="
date
