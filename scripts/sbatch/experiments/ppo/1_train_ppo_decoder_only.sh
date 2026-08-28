#!/bin/bash
set -e
#SBATCH --job-name=1_train_ppo_decoder_only
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/ppo/decoder_only/%x_%j.out
#SBATCH --error=results/logs/experiments/ppo/decoder_only/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


mkdir -p results/logs/experiments/ppo/decoder_only results/plots/experiments/ppo/decoder_only results/models/decoder_only/ppo results/evaluation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Starting Decoder-Only PPO Training (Qwen 2.5 1.5B) ==="
date

srun python scripts/modeling/decoder_only/train_ppo_decoder_only.py \
    --corpus_path "data/analysis/corpus_master.json" \
    --sft_model_path "results/models/decoder_only/sft" \
    --base_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --output_dir "results/models/decoder_only/ppo" \
    --log_dir "results/logs/experiments/ppo/decoder_only" \
    --plot_dir "results/plots/experiments/ppo/decoder_only" \
    --epochs 3 \
    --ppo_epochs 3 \
    --batch_size 4 \
    --mini_batch_size 2 \
    --lr 1e-5 \
    --vf_lr 3e-5 \
    --kl_beta 0.05 \
    --clip_eps 0.2 \
    --vf_coef 0.5 \
    --entropy_coef 0.01 \
    --max_source_len 512 \
    --max_target_len 256 \
    --temperature 0.7 \
    --top_p 0.9 \
    --repetition_penalty 1.2 \
    --w_style 0.5 \
    --w_sem 0.5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05

echo "=== Decoder-Only PPO Training Complete ==="
date
