#!/bin/bash
#SBATCH --job-name=3_create_lebenshilfe_dataset
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

srun python scripts/preprocessing/create_lebenshilfe_dataset.py \
    --data-dir data/texts_lebenshilfe
