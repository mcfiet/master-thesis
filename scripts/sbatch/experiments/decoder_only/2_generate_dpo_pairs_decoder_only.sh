#!/bin/bash
#SBATCH --job-name=2_generate_dpo_pairs_decoder_only
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_48gb:1
#SBATCH --output=results/logs/experiments/decoder_only/%x_%j.out
#SBATCH --error=results/logs/experiments/decoder_only/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi



mkdir -p results/logs/experiments/decoder_only results/plots/experiments/decoder_only results/evaluation data/dpo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Starting Decoder-Only DPO Dataset Generation (48GB GPU, Jina Embeddings) ==="
date

srun python scripts/modeling/decoder_only/generate_dpo_dataset_decoder_only.py \
    --corpus_path "data/corpus/corpus_10kgnad_len512_as.json" \
    --sft_model_path "results/models/decoder_only/sft" \
    --base_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --temperature_ladder 0.6 0.7 0.8 0.85 \
    --candidates_per_step 3 \
    --max_total_candidates 12 \
    --repetition_penalty 1.35 \
    --no_repeat_ngram_size 3 \
    --top_p 0.92 \
    --top_k 50 \
    --min_score_margin 0.05 \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --reward_max_seq_len 512 \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --w_style 0.7 \
    --w_sem 0.3 \
    --max_source_len 512 \
    --max_target_len 512 \
    --batch_size 16 \
    --output_file "data/dpo/dpo_preference_pairs_decoder_only.jsonl" \
    --val_split_ratio 0.15 \
    --resume

echo "=== DPO Dataset Generation Complete ==="
date

