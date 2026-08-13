#!/bin/bash
#SBATCH --job-name=1_filter_similarity
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:full:1

srun python scripts/preprocessing/1_filter_similarity.py \
    --analysis_csv data/analysis/information_loss_analysis.csv \
    --source_dir data/corpus/2_raw_scraped \
    --output_dir data/corpus/3_filtered_similarity \
    --sim_min 0.70 --sim_max 0.99 --min_ls_tokens 10
