#!/bin/bash
#SBATCH --job-name=06_prepare_10kgnad_dpo_corpus
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

mkdir -p data/corpus
mkdir -p results/logs

echo "=== Vorbereitung des ungesehenen 10kGNAD Alltagssprache-Korpus für DPO ==="
date

srun python scripts/data/prepare_10kgnad_corpus.py \
    --output_file "data/corpus/corpus_10kgnad_len500_as.json" \
    --min_tokens 50 \
    --max_tokens 500

echo "=== 10kGNAD Vorbereitung abgeschlossen ==="
date
