#!/bin/bash
#SBATCH --job-name=1_gen_dpo_w10_w00
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/metric_weights/%x_%j.out
#SBATCH --error=results/logs/experiments/metric_weights/%x_%j.err

mkdir -p data/metric_weights_exp
mkdir -p results/logs/experiments/metric_weights results/plots/experiments/metric_weights results/evaluation

echo "=== Generating DPO Preference Pairs (w_style=1.0, w_sem=0.0) ==="
date

srun python scripts/modeling/generate_dpo_dataset.py \
    --corpus_path "data/analysis/corpus_master.json" \
    --min_sim 0.70 \
    --max_sim 0.98 \
    --sft_model_path "results/models/sft" \
    --prompt_prefix "" \
    --num_candidates 5 \
    --temperature 0.8 \
    --max_source_len 256 \
    --max_target_len 256 \
    --reward_max_seq_len 256 \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --w_style 1.0 \
    --w_sem 0.0 \
    --min_score_margin 0.05 \
    --output_file "data/metric_weights_exp/dpo_pairs_w10_w00.jsonl" \
    --val_split_ratio 0.15

echo "=== Generation Completed ==="
date
