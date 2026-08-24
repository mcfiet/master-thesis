#!/bin/bash
#SBATCH --job-name=10_build_corpus_master
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:mig_24gb:1
#SBATCH --output=results/logs/run_pipeline/%x_%j.out
#SBATCH --error=results/logs/run_pipeline/%x_%j.err


mkdir -p results/logs/run_pipeline results/plots/run_pipeline results/evaluation
srun python scripts/preprocessing/build_corpus_master.py     --input_dir data/corpus/2_raw_scraped     --output_csv data/analysis/corpus_master.csv     --clean_json_dir data/corpus/4_normalized_clean
