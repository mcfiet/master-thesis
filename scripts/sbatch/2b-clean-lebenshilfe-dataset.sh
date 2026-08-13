#!/bin/bash
#SBATCH --job-name=2b_clean_lebenshilfe_dataset
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

srun python scripts/preprocessing/2b_clean_lebenshilfe.py
