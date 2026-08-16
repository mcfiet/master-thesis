#!/bin/bash
#SBATCH --job-name=4_clean_lebenshilfe
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

srun python scripts/preprocessing/clean_lebenshilfe.py \
    --input_file data/lebenshilfe/lebenshilfe_dataset.json \
    --output_file data/lebenshilfe/lebenshilfe_dataset_clean.json
