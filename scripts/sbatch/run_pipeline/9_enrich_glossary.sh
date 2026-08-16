#!/bin/bash
#SBATCH --job-name=9_enrich_glossary
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

srun python scripts/preprocessing/enrich_glossary.py
