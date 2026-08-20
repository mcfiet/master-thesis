#!/bin/bash
#SBATCH --job-name=0_prep_10kgnad_500
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

mkdir -p data/temperature_ladder_500
mkdir -p results/logs

echo "=== 1. Preparing 10kGNAD Alltagssprache Corpus (500 Tokens) ==="
date

srun python scripts/data/prepare_10kgnad_corpus.py \
    --output_file "data/temperature_ladder_500/corpus_10kgnad_len500_as.json" \
    --min_tokens 50 \
    --max_tokens 500 \

echo "=== Corpus Preparation Complete ==="
date
