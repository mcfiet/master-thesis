#!/bin/bash
#SBATCH --job-name=7_normalize_clean
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

srun python scripts/preprocessing/normalize_clean.py \
    --input_dir data/corpus/3_filtered_similarity \
    --output_dir data/corpus/4_normalized_clean
