#!/bin/bash
#SBATCH --job-name=1_crawl_url_alignment
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/logs/run_pipeline/%x_%j.out
#SBATCH --error=results/logs/run_pipeline/%x_%j.err


mkdir -p results/logs/run_pipeline results/plots/run_pipeline results/evaluation
echo "=== Schritt 1: URL-Alignment (12 Scraper) gestartet ==="

scrapers=(
    "apotheken_scraper.py"
    "behindertenbeauftragter_scraper.py"
    "brandeins_scraper.py"
    "hamburg_scraper.py"
    "hannover_scraper.py"
    "koeln_scraper.py"
    "main_taunus_scraper.py"
    "mdr_scraper.py"
    "sozialpolitik_scraper.py"
    "stuttgart_scraper.py"
    "taz_scraper.py"
    "wiesbaden_scraper.py"
)

for scraper in "${scrapers[@]}"; do
    echo "Starte: scripts/data_collection/crawl_scraper/$scraper"
    srun python "scripts/data_collection/crawl_scraper/$scraper"
done

echo "=== Schritt 1: URL-Alignment abgeschlossen ==="
