#!/bin/bash
#SBATCH --job-name=12_generate_synthetic_steps_corpus
#SBATCH --partition=research
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

srun python -u scripts/preprocessing/generate_synthetic_steps.py \
    --input data/analysis/corpus_master.json \
    --output data/corpus/corpus_master_with_steps.json \
    --url http://193.175.180.196:8000/v1/chat/completions \
    --token RrI6y403jAlUm8v \
    --model "FlensGen-GPT-OSS-120B"
