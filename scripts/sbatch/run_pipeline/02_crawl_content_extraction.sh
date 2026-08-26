#!/bin/bash
#SBATCH --job-name=02_crawl_content_extraction
#SBATCH --partition=research
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/logs/run_pipeline/%x_%j.out
#SBATCH --error=results/logs/run_pipeline/%x_%j.err

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi


mkdir -p results/logs/run_pipeline results/plots/run_pipeline results/evaluation
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

echo "=== Schritt 2: Content-Extraction (12 Scraper) gestartet ==="

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
    echo "Starte: scripts/data_collection/corpus_scrapers/$scraper"
    srun python "scripts/data_collection/corpus_scrapers/$scraper"
done

echo "=== Schritt 2: Content-Extraction abgeschlossen ==="
