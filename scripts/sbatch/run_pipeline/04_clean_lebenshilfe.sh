#!/bin/bash
#SBATCH --job-name=4_clean_lebenshilfe
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/logs/run_pipeline/%x_%j.out
#SBATCH --error=results/logs/run_pipeline/%x_%j.err


mkdir -p results/logs/run_pipeline results/plots/run_pipeline results/evaluation
srun python scripts/preprocessing/clean_lebenshilfe.py \
    --input_file data/lebenshilfe/lebenshilfe_dataset.json \
    --output_file data/lebenshilfe/lebenshilfe_dataset_clean.json
