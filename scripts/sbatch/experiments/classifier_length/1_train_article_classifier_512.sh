#!/bin/bash
#SBATCH --job-name=train_art_clf_512
#SBATCH --partition=research
#SBATCH --time=04:00:00
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
mkdir -p results/models/classifier_length_exp data/classifier_length_exp
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

srun python scripts/modeling/binary_train_article_model.py \
    --csv_path data/analysis/corpus_master.csv \
    --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --batch_size 32 \
    --embedding_dim 128 \
    --epochs 60 \
    --patience 15 \
    --hidden_dim 128 \
    --lr 0.001 \
    --max_seq_len 512 \
    --min_sim 0.80 \
    --max_sim 0.98 \
    --min_sent_len 4 \
    --model_save_path results/models/classifier_length_exp/bilstm_article_classifier_512.pt \
    --vocab_save_path data/classifier_length_exp/article_vocab_512.json \
    --log_dir results/logs/experiments/classifier_length \
    --plot_dir results/plots/experiments/classifier_length
