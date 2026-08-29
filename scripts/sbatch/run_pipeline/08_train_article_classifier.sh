#!/bin/bash
#SBATCH --job-name=08_train_article_classifier
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/run_pipeline/%x_%j.out
#SBATCH --error=results/logs/run_pipeline/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


mkdir -p results/logs/run_pipeline results/plots/run_pipeline results/evaluation
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

srun python scripts/modeling/binary_train_article_model.py \
    --csv_path data/analysis/corpus_master.csv \
    --lh_dataset_path data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --batch_size 32 \
    --embedding_dim 128 \
    --epochs 30 \
    --hidden_dim 128 \
    --lr 0.001 \
    --max_seq_len 256 \
    --max_sim 0.98 \
    --min_sent_len 3 \
    --min_sim 0.8 \
    --model_save_path results/models/bilstm_article_classifier.pt \
    --vocab_save_path data/vocabs/article_vocab.json
