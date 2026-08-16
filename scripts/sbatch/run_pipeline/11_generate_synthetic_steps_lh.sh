#!/bin/bash
#SBATCH --job-name=11_generate_synthetic_steps_lh
#SBATCH --partition=research
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

srun python -u scripts/preprocessing/generate_synthetic_steps.py \
    --input data/lebenshilfe/lebenshilfe_dataset_clean.json \
    --output data/lebenshilfe/lebenshilfe_dataset_with_steps.json \
    --url http://193.175.180.196:8000/v1/chat/completions \
    --token RrI6y403jAlUm8v \
    --model "FlensGen-GPT-OSS-120B"
