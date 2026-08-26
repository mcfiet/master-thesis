#!/bin/bash
#SBATCH --job-name=run_01_scraping
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/run_pipeline/%x_%j.out
#SBATCH --error=results/logs/run_pipeline/%x_%j.err
# =============================================================================
# Themenbereich 1: Web Scraping & Crawling
# =============================================================================
# Startet Schritt 01 (URL-Alignment) und Schritt 02 (Content-Extraktion)
# mit Slurm Job-Abhängigkeit (--dependency=afterok).
# =============================================================================

set -e

mkdir -p results/logs/run_pipeline
mkdir -p data/corpus/1_aligned_urls
mkdir -p data/corpus/2_raw_scraped

echo "========================================================================"
echo "Starte Themenbereich 1: Web Scraping & Crawling..."
echo "Gestartet am: $(date)"
echo "========================================================================"

JOB1=$(sbatch --parsable scripts/sbatch/run_pipeline/01_crawl_url_alignment.sh)
echo "Schritt 01 eingereicht (URL-Alignment): Job ID $JOB1"

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 scripts/sbatch/run_pipeline/02_crawl_content_extraction.sh)
echo "Schritt 02 eingereicht (Content-Extraktion): Job ID $JOB2"

echo "========================================================================"
echo "Scraping-Pipeline erfolgreich eingereicht!"
echo "Status prüfen mit: squeue -u \$USER"
echo "Logs überwachen in: results/logs/run_pipeline/"
echo "========================================================================"
