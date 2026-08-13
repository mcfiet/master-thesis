#!/bin/bash
#SBATCH --job-name=5_build_corpus_master
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:full:1

srun python scripts/preprocessing/5_build_corpus_master.py \
    --input_dir data/corpus/4_normalized_clean \
    --output_csv data/analysis/corpus_master.csv
