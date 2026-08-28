#!/bin/bash
#SBATCH --job-name=06_prepare_10kgnad_dpo_corpus
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/logs/run_pipeline/%x_%j.out
#SBATCH --error=results/logs/run_pipeline/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


mkdir -p data/corpus
mkdir -p results/logs/run_pipeline results/plots/run_pipeline results/evaluation
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU


echo "=== Vorbereitung des ungesehenen 10kGNAD Alltagssprache-Korpus für DPO ==="
date

srun python scripts/data/prepare_10kgnad_corpus.py \
    --output_file "data/corpus/corpus_10kgnad_len1024_as.json" \
    --min_tokens 50 \
    --max_tokens 1024

echo "=== 10kGNAD Vorbereitung abgeschlossen ==="
date
