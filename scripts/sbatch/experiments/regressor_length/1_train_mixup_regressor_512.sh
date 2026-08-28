#!/bin/bash
#SBATCH --job-name=train_mixup_512
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/experiments/regressor_length/%x_%j.out
#SBATCH --error=results/logs/experiments/regressor_length/%x_%j.err

set -e

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p results/logs/experiments/regressor_length results/plots/experiments/regressor_length results/evaluation
mkdir -p results/models/regressor_length_exp data/regressor_length_exp
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

srun python scripts/modeling/regression_train_mixup.py \
    --csv_path data/analysis/corpus_master.csv \
    --batch_size 64 \
    --embedding_dim 128 \
    --hidden_dim 128 \
    --lr 0.001 \
    --epochs 100 \
    --patience 20 \
    --max_seq_len 512 \
    --min_sim 0.80 \
    --max_sim 0.98 \
    --model_save_path results/models/regressor_length_exp/bilstm_mixup_regression_512.pt \
    --vocab_save_path data/regressor_length_exp/mixup_vocab_512.json \
    --log_dir results/logs/experiments/regressor_length \
    --plot_dir results/plots/experiments/regressor_length
