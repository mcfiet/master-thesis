# Experiment: Synthetischer Regressor

Dieses Experiment untersucht die Generierung von kontinuierlichen Komplexitäts-Zwischenstufen ($0.25, 0.50, 0.75$) mittels LLM-Prompting und das Training eines darauf basierenden BiLSTM-Regressors im empirischen Vergleich zum datengetriebenen MixUp-Ansatz (ohne nachgelagertes SFT/DPO-Alignment).

## Enthaltene Skripte:
* `generate_synthetic_steps.py`: Generiert Zwischenstufen zwischen Alltagssprache ($0.0$) und Leichter Sprache ($1.0$) über eine OpenAI-kompatible LLM-API.
* `regression_train_synthetic.py`: Trainiert ein BiLSTM-Regressionsmodell auf den synthetischen Textstufen.

## SBATCH-Ausführungsskripte:
Befinden sich unter `scripts/sbatch/experiments/metric/synthetic_regressor/`:
1. `1_generate_synthetic_steps_lh.sh`: Generierung synthetischer Zwischenstufen auf Lebenshilfe
2. `2_generate_synthetic_steps_corpus.sh`: Generierung synthetischer Zwischenstufen auf Corpus Master
3. `3_train_synthetic_regressor.sh`: Training des BiLSTM-Regressors auf den synthetischen Stufen
4. `4_evaluate_synthetic_experiments.sh`: Unbiased Stufen-Evaluation (MAE, MSE, Korrelationen)
5. `5_evaluate_synthetic_kde.sh`: Dichteverteilungen & Score-Separation auf Lebenshilfe
6. `run_all_synthetic_pipeline.sh`: Gesamter Ausführungs-Runner mit Slurm-Dependencies

