#!/bin/bash
#SBATCH --job-name=translate_custom_text
#SBATCH --partition=research
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=results/logs/experiments/translation/%x_%j.out
#SBATCH --error=results/logs/experiments/translation/%x_%j.err

set -e

# Virtuelle Python-Umgebung aktivieren
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "$HOME/master-thesis/.venv/bin/activate" ]; then
    source "$HOME/master-thesis/.venv/bin/activate"
fi

mkdir -p results/logs/experiments/translation results/translation
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU

echo "=== Starte benutzerdefinierte Textuebersetzung in Leichte Sprache ==="
date

# Standardwerte (koennen ueber Umgebungsvariablen oder CLI-Argumente ueberschrieben werden)
MODEL_PATH="${MODEL_PATH:-results/models/token_length_exp/dpo_len1024}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-facebook/mbart-large-50}"
INPUT_FILE="${INPUT_FILE:-data/experiments/translation/master_thesis_abstract.txt}"
OUTPUT_FILE="${OUTPUT_FILE:-results/translation/master_thesis_abstract_ls.txt}"

echo "Modell-Pfad : $MODEL_PATH"
echo "Basis-Modell: $BASE_MODEL_NAME"
echo "Eingabedatei: $INPUT_FILE"
echo "Ausgabedatei: $OUTPUT_FILE"

# Falls spezifische Argumente an das Skript uebergeben wurden, reiche diese 1:1 weiter,
# andernfalls nutze die Standard-Eingabedatei und automatische Device-Erkennung (CPU/GPU)
if [ "$#" -gt 0 ]; then
    srun python scripts/experiments/translate_custom_text.py "$@"
else
    srun python scripts/experiments/translate_custom_text.py \
        --model_path "$MODEL_PATH" \
        --base_model_name "$BASE_MODEL_NAME" \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE"
fi

echo "=== Uebersetzung erfolgreich abgeschlossen ==="
date
