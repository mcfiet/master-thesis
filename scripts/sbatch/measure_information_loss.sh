#!/bin/bash
#SBATCH --job-name=measure-information-loss
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:full:1

srun python scripts/evaluation/measure_information_loss.py \
    --input_dir data/corpus/2_raw_scraped \
    --output_csv data/analysis/information_loss_analysis.csv
