#!/bin/bash
#SBATCH --job-name=17b_generate_dpo_dataset_synthetic
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/experiments/synthetic_regressor/%x_%j.out
#SBATCH --error=results/logs/experiments/synthetic_regressor/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi



mkdir -p results/logs/experiments/synthetic_regressor results/plots/experiments/synthetic_regressor results/evaluation data/dpo
srun python scripts/modeling/generate_dpo_dataset.py \
    --corpus_path "data/corpus/corpus_10kgnad_len512_as.json" \
    --sft_model_path "results/models/experiments/synthetic_regressor/sft" \
    --base_model_name "facebook/mbart-large-50" \
    --temperature_ladder 0.6 0.7 0.8 0.85 \
    --candidates_per_step 3 \
    --max_total_candidates 12 \
    --repetition_penalty 1.35 \
    --no_repeat_ngram_size 3 \
    --max_source_len 256 \
    --max_target_len 256 \
    --reward_max_seq_len 256 \
    --reward_model_path "results/models/bilstm_synthetic_regression.pt" \
    --reward_vocab_path "data/vocabs/synthetic_vocab.json" \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --w_style 0.5 \
    --w_sem 0.5 \
    --min_score_margin 0.05 \
    --batch_size 16 \
    --output_file "data/dpo/dpo_preference_pairs_synthetic.jsonl" \
    --val_split_ratio 0.15 \
    --resume
