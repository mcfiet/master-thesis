#!/bin/bash
#SBATCH --job-name=1_train_ppo_seq2seq
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/ppo/seq2seq/%x_%j.out
#SBATCH --error=results/logs/experiments/ppo/seq2seq/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


mkdir -p results/logs/experiments/ppo/seq2seq results/plots/experiments/ppo/seq2seq results/models/ppo/seq2seq results/evaluation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Starting Seq2Seq PPO Training (mBART-50) ==="
date

srun python scripts/modeling/train_ppo_seq2seq.py \
    --corpus_path "data/analysis/corpus_master.json" \
    --sft_model_path "results/models/sft" \
    --base_model_name "facebook/mbart-large-50" \
    --reward_model_path "results/models/bilstm_mixup_regression.pt" \
    --reward_vocab_path "data/vocabs/mixup_vocab.json" \
    --sbert_model_name "jinaai/jina-embeddings-v2-base-de" \
    --output_dir "results/models/ppo/seq2seq" \
    --log_dir "results/logs/experiments/ppo/seq2seq" \
    --plot_dir "results/plots/experiments/ppo/seq2seq" \
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
    --max_source_len 256 \
    --max_target_len 256 \
    --temperature 0.7 \
    --top_p 0.9 \
    --repetition_penalty 1.2 \
    --w_style 0.5 \
    --w_sem 0.5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05

echo "=== Seq2Seq PPO Training Complete ==="
date
