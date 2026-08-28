#!/bin/bash
#SBATCH --job-name=run_all_pipeline_from_02
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/run_pipeline/%x_%j.out
#SBATCH --error=results/logs/run_pipeline/%x_%j.err

set -e
# =============================================================================
# Master Pipeline Runner: Ausführung ab Schritt 02 (Content-Extraktion)
# =============================================================================
# Startet die Master-Thesis-Pipeline ab Schritt 02 (Schritt 01 URL-Alignment
# wurde bereits ausgeführt und liegt vor).
# Automatische Job-Abhängigkeitskette (Slurm --dependency=afterok).
# =============================================================================


mkdir -p results/logs/run_pipeline
mkdir -p data/corpus
mkdir -p data/lebenshilfe
mkdir -p data/analysis
mkdir -p data/vocabs
mkdir -p results/models/sft
mkdir -p results/models/dpo
mkdir -p results/evaluation

echo "========================================================================"
echo "Starte Master-Pipeline Einreichung (ab Schritt 02)..."
echo "Gestartet am: $(date)"
echo "========================================================================"

echo "Schritt 01 übersprungen (URL-Alignment liegt bereits vor)."

# Stufe 1: Scraping & Lebenshilfe-Vorbereitung (Schritt 02 direkt starten)
JOB2=$(sbatch --parsable scripts/sbatch/run_pipeline/02_crawl_content_extraction.sh)
echo "Schritt 02 eingereicht (Content-Extraktion): Job ID $JOB2"

JOB3=$(sbatch --parsable scripts/sbatch/run_pipeline/03_create_lebenshilfe_dataset.sh)
echo "Schritt 03 eingereicht (Lebenshilfe einlesen): Job ID $JOB3"

JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 scripts/sbatch/run_pipeline/04_clean_lebenshilfe.sh)
echo "Schritt 04 eingereicht (Lebenshilfe bereinigen): Job ID $JOB4"

# Stufe 2: Master-Korpus & 10kGNAD DPO-Korpus bauen
JOB5=$(sbatch --parsable --dependency=afterok:$JOB2:$JOB4 scripts/sbatch/run_pipeline/05_build_corpus_master.sh)
echo "Schritt 05 eingereicht (Corpus Master erstellen): Job ID $JOB5"

JOB6=$(sbatch --parsable scripts/sbatch/run_pipeline/06_prepare_10kgnad_dpo_corpus.sh)
echo "Schritt 06 eingereicht (10kGNAD DPO-Korpus vorbereiten): Job ID $JOB6"

# Stufe 3: Klassifikatoren & MixUp-Regressor trainieren
JOB7=$(sbatch --parsable --dependency=afterok:$JOB5 scripts/sbatch/run_pipeline/07_train_sentence_classifier.sh)
echo "Schritt 07 eingereicht (Satz-Klassifikator): Job ID $JOB7"

JOB8=$(sbatch --parsable --dependency=afterok:$JOB5 scripts/sbatch/run_pipeline/08_train_article_classifier.sh)
echo "Schritt 08 eingereicht (Artikel-Klassifikator): Job ID $JOB8"

JOB9=$(sbatch --parsable --dependency=afterok:$JOB5 scripts/sbatch/run_pipeline/09_train_mixup_regressor.sh)
echo "Schritt 09 eingereicht (MixUp-Regressor): Job ID $JOB9"

# Stufe 4: SFT-Übersetzungsmodell trainieren
JOB10=$(sbatch --parsable --dependency=afterok:$JOB5:$JOB9 scripts/sbatch/run_pipeline/10_train_sft.sh)
echo "Schritt 10 eingereicht (SFT Training): Job ID $JOB10"

# Stufe 5: DPO-Präferenzpaare erzeugen (auf ungesehenem 10kGNAD mit SFT und Reward Model)
JOB11=$(sbatch --parsable --dependency=afterok:$JOB6:$JOB9:$JOB10 scripts/sbatch/run_pipeline/11_generate_dpo_dataset.sh)
echo "Schritt 11 eingereicht (DPO Paare generieren): Job ID $JOB11"

# Stufe 6: DPO-Training
JOB12=$(sbatch --parsable --dependency=afterok:$JOB10:$JOB11 scripts/sbatch/run_pipeline/12_train_dpo.sh)
echo "Schritt 12 eingereicht (DPO Training): Job ID $JOB12"

# Stufe 7: Finale Evaluierung auf dem Lebenshilfe Benchmark
JOB13=$(sbatch --parsable --dependency=afterok:$JOB4:$JOB10:$JOB12 scripts/sbatch/run_pipeline/13_evaluate_pipeline.sh)
echo "Schritt 13 eingereicht (Pipeline Evaluierung): Job ID $JOB13"

# Stufe 8: Experten-Evaluationspool erstellen (50 Items, 10 Nicht-Lebenshilfe Domaenen)
JOB_EXP=$(sbatch --parsable --dependency=afterok:$JOB9:$JOB10:$JOB12 scripts/sbatch/experiments/benchmark/2_run_expert_eval_benchmark.sh)
echo "Experten-Evaluationspool eingereicht (50 Items, verblindet): Job ID $JOB_EXP"

echo "========================================================================"
echo "Pipeline (ab Schritt 02) erfolgreich mit Job-Dependencies eingereicht!"
echo "Status prüfen mit: squeue -u \$USER"
echo "Logs überwachen in: results/logs/run_pipeline/"
echo "========================================================================"
