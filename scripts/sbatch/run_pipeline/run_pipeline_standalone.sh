#!/bin/bash
set -e

# Dummy-Funktion fängt 'srun' ab und führt nur die dahinterliegenden Argumente ("$@") aus
srun() {
    "$@"
}
export -f srun

mkdir -p results/logs/run_pipeline
mkdir -p data/corpus
mkdir -p data/lebenshilfe
mkdir -p data/analysis
mkdir -p data/vocabs
mkdir -p results/models/sft
mkdir -p results/models/dpo
mkdir -p results/evaluation

echo "========================================================================"
echo "Starte Master-Pipeline lokal (ab Schritt 02)..."
echo "Gestartet am: $(date)"
echo "========================================================================"

# Hilfsfunktion: Simuliert sbatch Logging (leitet stdout/stderr in Dateien um)
run_job() {
    local script=$1
    local job_name=$(basename "$script" .sh)
    # Nutze die aktuelle PID ($$) als Ersatz für die Slurm Job-ID (%j)
    local log_prefix="results/logs/run_pipeline/${job_name}_$$"
    
    echo "[$(date +'%H:%M:%S')] Starte $job_name ..."
    # Führt das Skript aus. (Entferne 'bash', falls die Skripte selbst ausführbar sind)
    bash "$script" > "${log_prefix}.out" 2> "${log_prefix}.err"
    echo "[$(date +'%H:%M:%S')] Abgeschlossen: $job_name"
}

# =============================================================================
# Stufe 1: Scraping & Vorbereitung (Parallel)
# =============================================================================
# Job 2 starten
run_job "scripts/sbatch/run_pipeline/02_crawl_content_extraction.sh" &
PID2=$!

# Job 6 ist unabhängig von 02-05, kann sofort im Hintergrund starten
run_job "scripts/sbatch/run_pipeline/06_prepare_10kgnad_dpo_corpus.sh" &
PID6=$!

# Job 3 und 4 müssen nacheinander laufen, aber parallel zu Job 2. 
# Wir klammern sie in eine Subshell.
(
    run_job "scripts/sbatch/run_pipeline/03_create_lebenshilfe_dataset.sh"
    run_job "scripts/sbatch/run_pipeline/04_clean_lebenshilfe.sh"
) &
PID_3_4=$!

# =============================================================================
# Stufe 2: Master-Korpus 
# =============================================================================
# Job 5 braucht 02 und 04 (welcher in der Subshell PID_3_4 liegt)
wait $PID2 $PID_3_4
run_job "scripts/sbatch/run_pipeline/05_build_corpus_master.sh"

# =============================================================================
# Stufe 3: Klassifikatoren & Regressor (Parallel)
# =============================================================================
# 07, 08 und 09 brauchen nur 05 (welcher gerade fertig wurde)
run_job "scripts/sbatch/run_pipeline/07_train_sentence_classifier.sh" &
run_job "scripts/sbatch/run_pipeline/08_train_article_classifier.sh" &
run_job "scripts/sbatch/run_pipeline/09_train_mixup_regressor.sh" &
PID9=$!

# =============================================================================
# Stufe 4-6: SFT & DPO (Linear)
# =============================================================================
# Job 10 braucht 05 (bereits gewartet) und 09
wait $PID9
run_job "scripts/sbatch/run_pipeline/10_train_sft.sh"

# Job 11 braucht 06, 09 (fertig) und 10 (gerade fertig)
wait $PID6
run_job "scripts/sbatch/run_pipeline/11_generate_dpo_dataset.sh"

# Job 12 braucht 10 und 11 (beide gerade fertig)
run_job "scripts/sbatch/run_pipeline/12_train_dpo.sh"

# =============================================================================
# Stufe 7-8: Finale Evaluierung (Parallel)
# =============================================================================
# 13 braucht 04, 10, 12 (alle bereits abgewartet)
run_job "scripts/sbatch/run_pipeline/13_evaluate_pipeline.sh" &

# EXP braucht 09, 10, 12 (alle bereits abgewartet)
run_job "scripts/sbatch/experiments/benchmark/2_run_expert_eval_benchmark.sh" &

# Warte auf alle noch laufenden Hintergrundprozesse (Klassifikatoren 07, 08 und die Evals)
wait

echo "========================================================================"
echo "Pipeline komplett abgeschlossen am: $(date)"
echo "Logs liegen in: results/logs/run_pipeline/"
echo "========================================================================"