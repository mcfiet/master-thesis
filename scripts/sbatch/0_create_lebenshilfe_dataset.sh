#!/bin/bash
#SBATCH --job-name=0_create_lebenshilfe_dataset
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

srun python scripts/preprocessing/0_create_lebenshilfe_dataset.py \ 
    --data-dir "data/lebenshilfe/texts_lebenshilfe"
